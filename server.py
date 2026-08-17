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
import uuid

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
# 비동기 job 저장소 — RunPod LB 는 요청 하나를 ~38초에 끊는다(실측). 화자분리는 15.8분에 27초,
# 60분이면 ~95초라 한 요청 안에 못 끝난다. POST 는 job_id 를 즉시 돌려주고 GET 으로 폴링한다.
# 워커 1대·job 은 짧게 살다 지워지므로 프로세스 메모리로 충분하다(워커가 죽으면 job 도 죽고
# BE 가 재시도한다).
_jobs: dict[str, dict] = {}
_JOB_TTL_SEC = 1800.0
_pipeline_lock = asyncio.Lock()
# ★ 추론은 한 번에 하나만. 파이프라인 객체가 하나(GPU 하나)라 동시 실행은 이득이 없고,
# to_thread 로 여러 추론이 겹치면 GPU·GIL 을 서로 밀어내 전부 느려진다(실측 2026-08-17:
# job 이 겹친 뒤 30초 오디오가 1초 → 17초, vLLM 전사까지 5.5초 → 30초로 끌려갔다).
_infer_lock = asyncio.Lock()
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
        # ⚠️ 무음 예열은 하지 않는다. v8 에서 5초 무음으로 첫 추론을 태웠더니 그 뒤 모든 추론이
        # **13배 느려졌다**(15.8분 27s → 346s, 2분 3s → 52s · 같은 코드에서 예열만 뺀 v7 은 빠름 —
        # 2026-08-17 클린 워커 A/B). 원인은 pyannote 가 첫 입력으로 내부 상태를 잡는 것으로
        # 보이며(빈 입력 → 퇴화), 첫 추론 페널티는 BE 의 예열(실오디오 30초)이 흡수한다.
    except Exception:  # noqa: BLE001
        log.exception("pyannote 예열 실패 — /health 가 503 을 유지한다")


@app.get("/diarize-stats")
async def diarize_stats():
    """진단용 — GPU 사용 여부·job 큐 상태. 콘솔 로그가 안 보이는 서버리스에서 이게 눈이다."""
    return {
        "cuda": torch.cuda.is_available(),
        "device": str(next(_pipeline._segmentation.model.parameters()).device) if _pipeline is not None else None,
        "jobs": {k: v.get("status") for k, v in _jobs.items()},
        "infer_locked": _infer_lock.locked(),
    }


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


def _diarize_sync(raw: bytes, num_speakers, min_speakers, max_speakers):
    """실제 화자분리 — 스레드에서 돈다. (segment, track, label) 3튜플 주의(pyannote 4)."""
    p = _pipeline
    assert p is not None
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
    ann = getattr(out, "exclusive_speaker_diarization", None)
    if ann is None:
        ann = getattr(out, "speaker_diarization", out)
    turns = [
        {"start": round(float(seg.start), 3), "end": round(float(seg.end), 3), "speaker": str(label)}
        for seg, _track, label in ann.itertracks(yield_label=True)
    ]
    return turns, waveform.shape[1] / sr


async def _run_job(job_id: str, raw: bytes, num_speakers, min_speakers, max_speakers):
    job = _jobs[job_id]
    t0 = time.time()
    try:
        await _get_pipeline()
        job["status"] = "queued"
        async with _infer_lock:
            job["status"] = "running"
            turns, dur = await asyncio.to_thread(_diarize_sync, raw, num_speakers, min_speakers, max_speakers)
        took = time.time() - t0
        job.update(status="done", turns=turns, duration=dur, took=round(took, 2), model=DIARIZE_MODEL)
        log.info("diarize job=%s %.0fs 오디오 → turn %d · 화자 %d · %.1fs (RTF %.3f)",
                 job_id, dur, len(turns), len({x["speaker"] for x in turns}), took, took / max(dur, 1e-6))
    except Exception as e:  # noqa: BLE001
        log.exception("diarize job=%s 실패", job_id)
        job.update(status="failed", error=f"{type(e).__name__}: {e}")
    finally:
        job["finished_at"] = time.time()


def _reap_jobs():
    now = time.time()
    for k in [k for k, j in _jobs.items() if j.get("finished_at") and now - j["finished_at"] > _JOB_TTL_SEC]:
        _jobs.pop(k, None)


@app.post("/diarize")
async def diarize_start(
    file: UploadFile = File(...),
    num_speakers: int | None = Form(None),
    min_speakers: int | None = Form(None),
    max_speakers: int | None = Form(None),
):
    """multipart 오디오 → **즉시** {job_id}. 결과는 GET /diarize/{job_id} 로 폴링.

    업로더가 적은 참석자 수는 num_speakers 로 그대로 싣는다. 결과는
    exclusive_speaker_diarization — 겹치는 발화를 화자 하나로 정리한 출력이라 BE 의
    '조각 = 화자 하나' 청킹에 바로 쓸 수 있다."""
    _reap_jobs()
    raw = await file.read()
    job_id = uuid.uuid4().hex
    _jobs[job_id] = {"status": "running", "created_at": time.time()}
    asyncio.create_task(_run_job(job_id, raw, num_speakers, min_speakers, max_speakers))
    return {"job_id": job_id, "status": "running"}


@app.get("/diarize/{job_id}")
async def diarize_status(job_id: str):
    job = _jobs.get(job_id)
    if job is None:
        return JSONResponse({"error": "unknown job"}, status_code=404)
    return {"job_id": job_id, **{k: v for k, v in job.items() if k not in ("created_at", "finished_at")}}


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
