"""Qwen3-ASR-1.7B 를 OpenAI 호환 `POST /audio/transcriptions` 로 서빙하는 RunPod 워커.

★ 이 파일의 존재 이유: BE 는 이미 `POST {base}/audio/transcriptions` 한 곳으로만 전사를
보낸다(server/services/whisper_hybrid.py `_whisper_chunk_request`). 그 요청·응답 계약을
**그대로** 흉내내면 BE 는 base_url 만 바뀌고 전송 코드는 한 줄도 안 바뀐다. 그래서 여기서
RunPod 큐(`/runsync` + `{"input": ...}`) 대신 **Load Balancing 엔드포인트**를 쓴다 —
워커의 HTTP 서버로 직결되어 커스텀 경로가 그대로 살아난다.

요청(BE 가 실제로 보내는 것):
    {"model": ..., "input_audio": {"data": <base64>, "format": "ogg"|"m4a"|"wav"},
     "language": "ko", "temperature": 0,                       # word_timestamps 일 때만
     "response_format": "verbose_json", "timestamp_granularities": ["word"]}

응답(BE 가 실제로 읽는 것):
    word_timestamps=False → {"text": ...}
    word_timestamps=True  → {"words": [{"word", "start", "end"}, ...]}
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import logging
import os
import subprocess
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse, Response

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s [stt-worker] %(message)s"
)
logger = logging.getLogger("stt-worker")

SR = 16_000  # Qwen3-ASR 입력 샘플레이트. 그 외는 ffmpeg 가 여기로 맞춘다.

MODEL_PATH = os.getenv("QWEN_ASR_MODEL", "Qwen/Qwen3-ASR-1.7B")
ALIGNER_PATH = os.getenv("QWEN_ALIGNER_MODEL", "Qwen/Qwen3-ForcedAligner-0.6B")
BACKEND = os.getenv("QWEN_ASR_BACKEND", "vllm")  # vllm | transformers
GPU_MEM_UTIL = float(os.getenv("QWEN_GPU_MEMORY_UTILIZATION", "0.85"))
MAX_NEW_TOKENS = int(os.getenv("QWEN_MAX_NEW_TOKENS", "4096"))

# 마이크로 배칭 — 자체 호스팅의 실익이 여기서 난다. BE 는 300초 격자 조각을 P 병렬로
# 던지므로, 도착한 것들을 한 번에 묶어 vLLM 한 패스로 태우면 GPU 점유 시간이 크게 준다.
# BATCH_WAIT_MS 는 "첫 요청이 도착한 뒤 동료를 기다리는 시간" 이다 — 0 이면 배칭 없음.
MAX_BATCH = int(os.getenv("QWEN_MAX_BATCH", "8"))
BATCH_WAIT_MS = int(os.getenv("QWEN_BATCH_WAIT_MS", "150"))

# 선택적 공유 비밀. RunPod 앞단 인증과 별개로, 엔드포인트 URL 이 새더라도 남이 못 쓰게 한다.
API_KEY = os.getenv("STT_API_KEY", "")

_model: Any = None
_ready = False
_load_error: str | None = None
_queue: asyncio.Queue["_Job"] | None = None
_shape_logged = False


# ───────────────────────────── 오디오 디코딩 ─────────────────────────────


def _decode_audio(raw: bytes) -> np.ndarray:
    """임의 컨테이너(ogg/opus·m4a·wav) → 16kHz mono float32.

    ★ soundfile 대신 ffmpeg 를 쓰는 이유: BE 는 컨테이너를 설정으로 바꾼다
    (`openrouter_stt_audio_format`). libsndfile 은 빌드에 따라 opus·m4a 지원이 갈리는데,
    포맷을 잘못 읽으면 **에러가 아니라 빈 오디오**로 나가 전사문이 조용히 비는 사고가 된다
    (BE 주석의 chirp-3 m4a 사고와 같은 종류). ffmpeg 는 셋 다 확실히 읽는다."""
    process = subprocess.run(
        # fmt: off
        ["ffmpeg", "-hide_banner", "-loglevel", "error",
         "-i", "pipe:0",
         "-f", "f32le", "-acodec", "pcm_f32le", "-ar", str(SR), "-ac", "1",
         "pipe:1"],
        # fmt: on
        input=raw,
        capture_output=True,
        check=False,
    )
    if process.returncode != 0:
        detail = process.stderr.decode("utf-8", "replace")[:300]
        raise HTTPException(status_code=400, detail=f"오디오 디코딩 실패: {detail}")
    samples = np.frombuffer(process.stdout, dtype=np.float32)
    if samples.size == 0:
        raise HTTPException(status_code=400, detail="오디오 디코딩 결과가 비었습니다")
    return samples


# ───────────────────────────── 모델 로딩 ─────────────────────────────


def _load_model() -> Any:
    from qwen_asr import Qwen3ASRModel

    if BACKEND == "vllm":
        logger.info("vLLM 백엔드로 %s 로딩 (aligner=%s)", MODEL_PATH, ALIGNER_PATH)
        return Qwen3ASRModel.LLM(
            model=MODEL_PATH,
            forced_aligner=ALIGNER_PATH,
            gpu_memory_utilization=GPU_MEM_UTIL,
            max_new_tokens=MAX_NEW_TOKENS,
        )

    import torch

    logger.info("transformers 백엔드로 %s 로딩 (aligner=%s)", MODEL_PATH, ALIGNER_PATH)
    return Qwen3ASRModel.from_pretrained(
        MODEL_PATH,
        forced_aligner=ALIGNER_PATH,
        dtype=torch.bfloat16,
        device_map="cuda:0",
        max_inference_batch_size=MAX_BATCH,
        max_new_tokens=MAX_NEW_TOKENS,
    )


# ───────────────────────────── 배치 실행 ─────────────────────────────


@dataclass
class _Job:
    audio: np.ndarray
    language: str | None
    want_timestamps: bool
    future: asyncio.Future = field(default_factory=asyncio.Future)


def _normalize_words(entries: Any, duration: float) -> list[dict[str, Any]]:
    """정렬기 출력 → BE 가 읽는 `[{word, start, end}]`.

    ⚠️ `time_stamps` 의 필드 이름은 qwen-asr 판올림에 따라 갈릴 수 있어(word/text/token,
    start/begin) 흔한 이름을 모두 받아준다. 모르는 모양이 오면 조용히 빈 목록을 주는 대신
    첫 1건을 통째로 로그에 남긴다 — 빈 전사문은 원인을 못 찾는 사고라서다."""
    global _shape_logged
    words: list[dict[str, Any]] = []
    for entry in entries or []:
        if isinstance(entry, dict):
            source = entry
        else:  # dataclass·pydantic 등 속성 객체
            source = {
                key: getattr(entry, key)
                for key in ("word", "text", "token", "start", "end", "begin", "start_time", "end_time")
                if hasattr(entry, key)
            }
        token = str(source.get("word") or source.get("text") or source.get("token") or "").strip()
        if not token:
            continue
        start = source.get("start", source.get("begin", source.get("start_time")))
        end = source.get("end", source.get("end_time"))
        try:
            start_f = float(start)
            end_f = float(end)
        except (TypeError, ValueError):
            continue
        words.append({"word": token, "start": start_f, "end": min(end_f, duration)})

    if not words and entries and not _shape_logged:
        _shape_logged = True
        logger.error("time_stamps 모양을 해석하지 못했습니다 — 원본 첫 건: %r", entries[0])
    return words


def _run_batch(jobs: list[_Job]) -> list[tuple[str, list[dict[str, Any]]]]:
    """블로킹 추론 1회. 배치 안의 언어·타임스탬프 요구가 섞이면 안 되므로 호출자가 갈라 준다."""
    want_timestamps = jobs[0].want_timestamps
    results = _model.transcribe(
        audio=[(job.audio, SR) for job in jobs],
        language=jobs[0].language,
        return_time_stamps=want_timestamps,
    )
    output: list[tuple[str, list[dict[str, Any]]]] = []
    for job, result in zip(jobs, results, strict=True):
        text = str(getattr(result, "text", "") or "")
        duration = len(job.audio) / SR
        words = (
            _normalize_words(getattr(result, "time_stamps", None), duration)
            if want_timestamps
            else []
        )
        output.append((text, words))
    return output


async def _batch_loop() -> None:
    """도착한 요청을 MAX_BATCH 까지 모아 한 패스로 태운다.

    ★ 추론을 이 루프 하나로 직렬화한다 — vLLM 엔진에 여러 스레드가 동시에 들어가면 VRAM
    사용이 예측 불가해지고, RunPod 서버리스는 OOM 이 곧 워커 사망이라 비용으로 돌아온다."""
    assert _queue is not None
    while True:
        first = await _queue.get()
        batch = [first]
        if BATCH_WAIT_MS > 0:
            deadline = time.monotonic() + BATCH_WAIT_MS / 1000
            while len(batch) < MAX_BATCH:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    candidate = await asyncio.wait_for(_queue.get(), timeout=remaining)
                except asyncio.TimeoutError:
                    break
                # 언어·타임스탬프 요구가 다르면 같은 패스에 못 태운다 — 되돌려 놓는다.
                if (
                    candidate.want_timestamps != first.want_timestamps
                    or candidate.language != first.language
                ):
                    _queue.put_nowait(candidate)
                    break
                batch.append(candidate)

        pending = [job for job in batch if not job.future.done()]
        if not pending:
            continue
        try:
            results = await asyncio.to_thread(_run_batch, pending)
        except Exception as error:  # noqa: BLE001 — 배치 하나의 실패로 루프를 죽이지 않는다
            logger.exception("배치 추론 실패 (%d건)", len(pending))
            for job in pending:
                if not job.future.done():
                    job.future.set_exception(error)
            continue
        for job, result in zip(pending, results, strict=True):
            if not job.future.done():
                job.future.set_result(result)


# ───────────────────────────── FastAPI ─────────────────────────────


@asynccontextmanager
async def lifespan(_: FastAPI):
    global _model, _ready, _load_error, _queue
    _queue = asyncio.Queue()
    loop_task = asyncio.create_task(_batch_loop())
    try:
        started = time.monotonic()
        _model = await asyncio.to_thread(_load_model)
        _ready = True
        logger.info("모델 로딩 완료 — %.1f초", time.monotonic() - started)
    except Exception as error:  # noqa: BLE001
        # ★ 여기서 죽지 않는다. 죽으면 RunPod 이 워커를 재시작만 반복해 원인이 안 보인다.
        # /ping 이 503 을 내 unhealthy 로 표시되고, 로그에 사유가 남는다.
        _load_error = f"{type(error).__name__}: {error}"
        logger.exception("모델 로딩 실패")
    try:
        yield
    finally:
        loop_task.cancel()


app = FastAPI(lifespan=lifespan)


@app.get("/ping")
async def ping() -> Response:
    """RunPod 헬스체크. 200=정상 · 204=초기화 중 · 그 외=비정상."""
    if _load_error:
        return JSONResponse({"error": _load_error}, status_code=503)
    return Response(status_code=200 if _ready else 204)


@app.post("/audio/transcriptions")
async def transcriptions(request: Request, authorization: str = Header(default="")) -> dict[str, Any]:
    if API_KEY and authorization.removeprefix("Bearer ").strip() != API_KEY:
        raise HTTPException(status_code=401, detail="인증 실패")
    if not _ready:
        # 503 은 BE 의 TRANSPORT_RETRYABLE_STATUS 라 백오프 재시도로 흡수된다.
        raise HTTPException(status_code=503, detail=_load_error or "모델 로딩 중")

    body = await request.json()
    audio_field = body.get("input_audio") or {}
    encoded = audio_field.get("data")
    if not encoded:
        raise HTTPException(status_code=400, detail="input_audio.data 가 필요합니다")
    try:
        raw = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as error:
        raise HTTPException(status_code=400, detail="input_audio.data base64 디코딩 실패") from error

    audio = await asyncio.to_thread(_decode_audio, raw)
    duration = len(audio) / SR
    want_timestamps = "word" in (body.get("timestamp_granularities") or [])

    assert _queue is not None
    job = _Job(audio=audio, language=body.get("language"), want_timestamps=want_timestamps)
    _queue.put_nowait(job)
    text, words = await job.future

    logger.info(
        "전사 완료 — %.1f초 오디오, %d자, words=%d", duration, len(text), len(words)
    )
    payload: dict[str, Any] = {"text": text, "usage": {"seconds": round(duration)}}
    if want_timestamps:
        # BE 는 verbose_json 일 때 payload["words"] 만 읽는다.
        payload["words"] = words
    return payload


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "80")))
