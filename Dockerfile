# Qwen3-ASR-1.7B + pyannote 화자분리 RunPod 워커 — 공식 vLLM 이미지 위에 최소한만 얹는다.
#
# ★ 경위(2026-08-17). 처음에는 CUDA 베이스 위에 qwen-asr 패키지와 FastAPI 서버를 직접 쌓았다.
# 빌드는 통과했지만 **RunPod 워커가 로그 한 줄 없이 initializing 에서 못 나왔고 원인을 끝내
# 특정하지 못했다.** 공식 이미지는 같은 계정·같은 DC 에서 정상 기동했고 에러도 정확히 뱉었다.
# 그래서 **직접 쌓지 않고 공식 이미지에 최소한만 더한다** — vLLM 은 Qwen3-ASR 을 정식 지원하고
# OpenAI 호환 `/v1/audio/transcriptions` 를 그대로 준다.
#
# 2026-08-17 (2차): 데이터 반출 0 을 위해 화자분리(종전 pyannote.ai)도 같은 워커에 넣는다.
# vLLM 은 그대로 :8000 에 두고, 앞에 얇은 프록시(:80)를 세워 /diarize 만 우리가 처리하고
# 나머지는 vLLM 으로 넘긴다(server.py · start.sh). ENTRYPOINT 를 덮는 것이라 로컬에서
# `/health` 200 을 먼저 확인하고 올린다.
#
# ⚠️ 태그를 :latest 로 두는 이유: v0.14.0 은 `qwen3_asr` 을 몰라 기동에 실패한다. 0.27 대부터 지원.
FROM vllm/vllm-openai:latest

# 오디오 확장(없으면 전사가 `Please install vllm[audio]` 로 400) + pyannote + 프록시.
# torch/torchaudio 는 베이스 것을 쓴다 — 여기서 재설치하면 CUDA 빌드가 갈려 안 뜬다.
RUN pip install --no-cache-dir \
      librosa soundfile \
      "pyannote.audio>=4.0" \
      fastapi "uvicorn[standard]" python-multipart httpx

WORKDIR /app
COPY server.py start.sh /app/
RUN chmod +x /app/start.sh

# 모델은 굽지 않는다 — 가중치 다운로드는 콜드스타트의 17초뿐이고(실측), 구운 13.76GB 이미지는
# 워커를 못 뜨게 했다. HF 토큰은 RunPod 엔드포인트 env(HUGGING_FACE_HUB_TOKEN)로 넣는다.
ENV PORT=80
EXPOSE 80
ENTRYPOINT ["/app/start.sh"]
