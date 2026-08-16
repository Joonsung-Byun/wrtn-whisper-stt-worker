# Qwen3-ASR-1.7B RunPod Load Balancing 워커
#
# 베이스를 vllm/vllm-openai 로 잡은 이유: torch·CUDA·vLLM 이 이미 맞물려 들어가 있어
# nvidia/cuda 에서 직접 쌓는 것보다 빌드가 훨씬 짧고 버전 충돌이 없다. ENTRYPOINT 는
# vLLM 서버라 아래에서 비운다.
FROM vllm/vllm-openai:latest

ENTRYPOINT []

# ffmpeg — BE 가 보내는 ogg/opus·m4a 를 확실히 디코딩하려면 필요하다(server.py 주석 참고)
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ★ 가중치를 이미지에 굽는다. 서버리스는 콜드스타트가 곧 과금이라, 워커가 뜰 때마다
# HF 에서 ~8GB 를 받으면 첫 요청이 수십 초 늦고 그 시간도 청구된다. 이미지가 커지는 대신
# 워커 머신에 캐시되므로 두 번째 부팅부터는 로딩만 남는다.
# 굽지 않고 런타임에 받게 하려면 --build-arg BAKE_WEIGHTS=0 으로 끄고, RunPod 쪽에
# 네트워크 볼륨을 붙여 HF_HOME 을 그 위로 보낸다.
ARG BAKE_WEIGHTS=1
ENV HF_HOME=/models
RUN if [ "$BAKE_WEIGHTS" = "1" ]; then \
        huggingface-cli download Qwen/Qwen3-ASR-1.7B && \
        huggingface-cli download Qwen/Qwen3-ForcedAligner-0.6B ; \
    fi

COPY server.py .

# RunPod Load Balancing 은 PORT 를 주입한다(기본 80). HEALTH_CHECK_PATH 기본값 /ping 을
# server.py 가 그대로 구현한다.
ENV PORT=80
EXPOSE 80

CMD ["sh", "-c", "exec uvicorn server:app --host 0.0.0.0 --port ${PORT:-80}"]
