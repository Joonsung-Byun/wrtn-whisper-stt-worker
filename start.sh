#!/bin/bash
# 워커 기동 — vLLM 을 :8000 에 백그라운드로, 프록시(+pyannote)를 :80 에.
# RunPod "Container start command" 는 비워 둔다(이 스크립트가 ENTRYPOINT). 모델·옵션은 env 로.
set -euo pipefail

MODEL="${STT_MODEL:-Qwen/Qwen3-ASR-1.7B}"
PORT="${PORT:-80}"

echo "[start] vllm serve $MODEL → :8000"
# 종전 RunPod 시작 명령을 그대로 옮겼다 (--enforce-eager 가 콜드 761→176s 의 주역).
vllm serve "$MODEL" \
  --host 127.0.0.1 --port 8000 \
  --gpu-memory-utilization "${GPU_MEM_UTIL:-0.80}" \
  --max-model-len "${MAX_MODEL_LEN:-8192}" \
  --enforce-eager \
  ${VLLM_EXTRA_ARGS:-} &
VLLM_PID=$!

# vLLM 이 죽으면 컨테이너도 죽는다 — 반쯤 살아서 LB 에 붙잡히는 상태를 만들지 않는다.
( while kill -0 $VLLM_PID 2>/dev/null; do sleep 5; done; echo "[start] vllm 종료 — 컨테이너 내림"; kill -TERM $$ ) &

echo "[start] proxy+pyannote → :$PORT"
exec uvicorn server:app --host 0.0.0.0 --port "$PORT" --workers 1 --log-level info
