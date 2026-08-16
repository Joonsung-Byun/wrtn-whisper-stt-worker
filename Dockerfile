# Qwen3-ASR-1.7B RunPod Load Balancing 워커
#
# 베이스를 vllm/vllm-openai 로 잡은 이유: torch·CUDA·vLLM 이 이미 맞물려 들어가 있어
# nvidia/cuda 에서 직접 쌓는 것보다 빌드가 훨씬 짧고 버전 충돌이 없다. ENTRYPOINT 는
# vLLM 서버라 아래에서 비운다.
#
# ★ 태그를 :latest 가 아니라 **v0.14.0 으로 못박는다.** qwen-asr[vllm] 이 vllm==0.14.0 을
# 정확히 요구하므로, latest 를 쓰면 pip 가 vllm 을 갈아끼우면서 베이스의 CUDA 스택까지
# 건드린다(실측 2026-08-16 — nvidia-nccl-cu13 제거). 버전을 맞춰두면 pip 는 "이미 충족"
# 으로 넘어가고 torch·vllm 은 그대로 남는다. requirements.txt 의 핀과 **짝으로** 올린다.
FROM vllm/vllm-openai:v0.14.0

ENTRYPOINT []

# ffmpeg — BE 가 보내는 ogg/opus·m4a 를 확실히 디코딩하려면 필요하다(server.py 주석 참고)
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
# ⚠️ `--ignore-installed blinker` — 베이스의 blinker 1.4 는 OS 패키지(distutils)로 깔려 있어
# pip 가 "어떤 파일이 이 패키지 것인지 알 수 없다"며 제거를 거부하고 설치 전체를 실패시킨다
# (실측 2026-08-16). qwen-asr 가 flask 를 필수 의존성으로 끌고 오는 한 이 충돌은 피할 수 없어,
# 해당 패키지만 덮어쓰기로 지나간다.
RUN pip install --no-cache-dir --ignore-installed blinker -r requirements.txt

# ★ 가중치를 이미지에 굽는다. 서버리스는 콜드스타트가 곧 과금이라, 워커가 뜰 때마다
# HF 에서 ~8GB 를 받으면 첫 요청이 수십 초 늦고 그 시간도 청구된다. 이미지가 커지는 대신
# 워커 머신에 캐시되므로 두 번째 부팅부터는 로딩만 남는다.
# 굽지 않고 런타임에 받게 하려면 --build-arg BAKE_WEIGHTS=0 으로 끄고, RunPod 쪽에
# 네트워크 볼륨을 붙여 HF_HOME 을 그 위로 보낸다.
ARG BAKE_WEIGHTS=1
ENV HF_HOME=/models
# ⚠️ 다운로드 CLI 이름이 판올림 중이다 — 최신 huggingface_hub 은 `hf`, 구버전은
# `huggingface-cli` 다. 둘 중 있는 쪽을 쓴다(한쪽만 박으면 베이스 이미지 갱신에 깨진다).
RUN if [ "$BAKE_WEIGHTS" = "1" ]; then \
        if command -v hf >/dev/null 2>&1; then DL="hf download"; \
        else DL="huggingface-cli download"; fi && \
        $DL Qwen/Qwen3-ASR-1.7B && \
        $DL Qwen/Qwen3-ForcedAligner-0.6B && \
        du -sh /models ; \
    fi

COPY server.py .

# RunPod Load Balancing 은 PORT 를 주입한다(기본 80). HEALTH_CHECK_PATH 기본값 /ping 을
# server.py 가 그대로 구현한다.
ENV PORT=80
EXPOSE 80

CMD ["sh", "-c", "exec uvicorn server:app --host 0.0.0.0 --port ${PORT:-80}"]
