"""RunPod 워커 프론트 — 한 포트(:80)에 화자분리와 vLLM 을 함께 노출한다.

  POST /diarize            pyannote (이 프로세스, GPU) — 화자 turn 만 돌려준다
  GET  /health             둘 다 준비됐을 때만 200 (RunPod LB 가 이걸 본다)
  나머지 (/v1/* 등)         vLLM(:8000)으로 그대로 전달 — 전사 API 는 vLLM 이 소유

왜 이 모양인가: 데이터 반출 0 이 1순위라 화자분리(종전 pyannote.ai)도 같은 워커에 넣는다.
GPU 는 24GB 중 qwen 이 ~4GB 만 써서 pyannote(~1GB)가 같이 살고, 콜드스타트도 한 번이다.
vLLM 은 건드리지 않는다 — 공식 이미지의 `vllm serve` 를 백그라운드로 띄우고 앞에 이 프록시만
얹는다(start.sh). 화자분리 자체 코드는 pyannote 파이프라인 호출 한 줄이라 서버가 얇다.
"""

from __future__ import annotations

import asyncio
import io
import logging
import os
import time

import httpx
import numpy as np
import soundfile as sf
import torch
from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import JSONResponse, Response

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("worker")

VLLM = os.environ.get("VLLM_URL", "http://127.0.0.1:8000")
DIARIZE_MODEL = os.environ.get("DIARIZE_MODEL", "pyannote/speaker-diarization-community-1")
HF_TOKEN = os.environ.get("HUGGING_FACE_HUB_TOKEN") or os.environ.get("HF_TOKEN")

app = FastAPI()
_pipeline = None
_pipeline_lock = asyncio.Lock()
_client = httpx.AsyncClient(base_url=VLLM, timeout=httpx.Timeout(600.0))


def _load_pipeline():
    """pyannote 파이프라인을 GPU 에 올린다. 무겁고(수십 초) 1회라 lazy + lock."""
    from pyannote.audio import Pipeline

    t0 = time.time()
    p = Pipeline.from_pretrained(DIARIZE_MODEL, token=HF_TOKEN)
    if p is None:
        raise RuntimeError(f"pyannote 파이프라인 로드 실패 — 토큰·약관 동의 확인: {DIARIZE_MODEL}")
    p.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    log.info("pyannote 로드 %.1fs · %s · cuda=%s", time.time() - t0, DIARIZE_MODEL, torch.cuda.is_available())
    return p


async def _get_pipeline():
    global _pipeline
    if _pipeline is None:
        async with _pipeline_lock:
            if _pipeline is None:
                _pipeline = await asyncio.to_thread(_load_pipeline)
    return _pipeline


@app.on_event("startup")
async def _warm():
    # 기동 시 미리 올린다 — 첫 /diarize 가 모델 로드까지 기다리지 않게. 실패해도 서버는 뜨고
    # /health 가 not-ready 를 돌려주므로 LB 가 트래픽을 안 흘린다.
    try:
        await _get_pipeline()
    except Exception:  # noqa: BLE001
        log.exception("pyannote 예열 실패 — /health 가 503 을 유지한다")


@app.get("/health")
async def health():
    """둘 다 준비돼야 200. RunPod LB 는 이 응답으로 트래픽을 흘릴지 정한다."""
    if _pipeline is None:
        return JSONResponse({"status": "diarize not ready"}, status_code=503)
    try:
        r = await _client.get("/health", timeout=5.0)
        if r.status_code != 200:
            return JSONResponse({"status": f"vllm {r.status_code}"}, status_code=503)
    except httpx.HTTPError as e:
        return JSONResponse({"status": f"vllm unreachable: {type(e).__name__}"}, status_code=503)
    return {"status": "ok"}


@app.post("/diarize")
async def diarize(
    file: UploadFile = File(...),
    num_speakers: int | None = Form(None),
    min_speakers: int | None = Form(None),
    max_speakers: int | None = Form(None),
):
    """multipart 오디오 → 화자 turn 목록. 업로더가 적은 참석자 수는 num_speakers 로 그대로 싣는다.

    exclusive_speaker_diarization 을 돌려준다 — 겹치는 발화를 화자 하나로 정리한 출력이라
    BE 의 '조각 = 화자 하나' 청킹에 바로 쓸 수 있다."""
    p = await _get_pipeline()
    raw = await file.read()
    t0 = time.time()

    def _run():
        # torchaudio.load 대신 soundfile — pyannote 4.x 가 끌고 오는 torchcodec 은 FFmpeg 공유
        # 라이브러리를 요구해 베이스 이미지에서 import 가 깨질 수 있다. soundfile 은 이미 있고
        # wav/flac/ogg 를 직접 읽는다(BE 는 FLAC 을 보낸다).
        data, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=True)  # (frames, ch)
        mono = data.mean(axis=1) if data.shape[1] > 1 else data[:, 0]
        waveform = torch.from_numpy(np.ascontiguousarray(mono)).unsqueeze(0)  # (1, frames)
        kwargs = {}
        if num_speakers:
            kwargs["num_speakers"] = num_speakers
        else:
            if min_speakers:
                kwargs["min_speakers"] = min_speakers
            if max_speakers:
                kwargs["max_speakers"] = max_speakers
        out = p({"waveform": waveform, "sample_rate": sr}, **kwargs)
        # 4.x 는 exclusive 를 주고, 혹시 없으면(구버전) 일반 diarization 으로 떨어진다.
        ann = getattr(out, "exclusive_speaker_diarization", None)
        if ann is None:
            ann = getattr(out, "speaker_diarization", out)
        turns = [
            {"start": round(float(t.start), 3), "end": round(float(t.end), 3), "speaker": str(s)}
            for t, s in ann.itertracks(yield_label=True)
        ]
        return turns, waveform.shape[1] / sr

    turns, dur = await asyncio.to_thread(_run)
    took = time.time() - t0
    log.info("diarize %.0fs 오디오 → turn %d · 화자 %d · %.1fs (RTF %.3f)",
             dur, len(turns), len({t['speaker'] for t in turns}), took, took / max(dur, 1e-6))
    return {"turns": turns, "duration": dur, "model": DIARIZE_MODEL, "took": round(took, 2)}


@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "HEAD"])
async def proxy(path: str, request: Request):
    """그 외 전부 vLLM 으로. 본문·헤더·상태·응답을 그대로 넘긴다."""
    body = await request.body()
    headers = {k: v for k, v in request.headers.items() if k.lower() not in ("host", "content-length")}
    r = await _client.request(request.method, f"/{path}", params=request.query_params, headers=headers, content=body)
    drop = {"content-encoding", "transfer-encoding", "connection", "content-length"}
    return Response(content=r.content, status_code=r.status_code,
                    headers={k: v for k, v in r.headers.items() if k.lower() not in drop},
                    media_type=r.headers.get("content-type"))
