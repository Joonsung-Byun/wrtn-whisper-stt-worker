# Qwen3-ASR-1.7B RunPod 워커 — 공식 vLLM 이미지 + 오디오 확장만 얹는다.
#
# ★ 여기까지 온 경위(2026-08-17). 처음에는 CUDA 베이스 위에 qwen-asr 패키지와 FastAPI 서버를
# 직접 쌓았다. 빌드는 결국 통과했지만 **RunPod 워커가 컨테이너 로그 한 줄 없이 initializing 에서
# 나오지 못했고 원인을 끝내 특정하지 못했다.** 반면 공식 이미지는 같은 계정·같은 데이터센터에서
# 정상 기동했고, 에러도 정확히 뱉었다(명령어 오류 → 모델 미지원 → 오디오 확장 누락). 그래서
# **직접 쌓지 않고 공식 이미지에 최소한만 더한다** 는 방침으로 되돌렸다.
#
# vLLM 은 Qwen3-ASR 을 정식 지원하며(qwen3_asr 모델 코드 내장) OpenAI 호환
# `/v1/audio/transcriptions` 를 그대로 제공한다 — 즉 우리가 서버를 짤 이유가 없다.
#
# ⚠️ 태그를 :latest 로 두는 이유: v0.14.0 은 `qwen3_asr` 아키텍처를 몰라 기동에 실패한다
# (실측 — "Transformers does not recognize this architecture"). 지원이 들어간 것은 0.27 대다.
FROM vllm/vllm-openai:latest

# ★ 이 이미지의 유일한 존재 이유.
# 공식 이미지는 오디오 확장을 빼고 배포된다 — 전사 요청을 보내면 모델·서버가 멀쩡한데도
# `Failed to load audio via soundfile: ImportError('Please install vllm[audio] for audio support')`
# 로 400 이 난다(실측 2026-08-17). librosa·soundfile 만 얹으면 해소된다.
RUN pip install --no-cache-dir librosa soundfile

# 베이스의 ENTRYPOINT(`vllm serve`)를 그대로 물려받는다 — RunPod 의 "Container start command"
# 에는 **모델명과 옵션만** 넣는다. `vllm serve` 를 다시 쓰면 인자가 겹쳐 기동에 실패한다(실측).
#   예: Qwen/Qwen3-ASR-1.7B --host 0.0.0.0 --port 80 --gpu-memory-utilization 0.85
