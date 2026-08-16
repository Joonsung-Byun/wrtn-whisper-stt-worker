# Qwen3-ASR-1.7B RunPod Load Balancing 워커
#
# ★ 베이스는 CUDA **runtime** 최소 이미지다. 처음에는 `vllm/vllm-openai` 를 썼는데
# (torch·CUDA·vLLM 이 이미 맞물려 있어 편했다) 압축 해제 시 25~30GB 라 **GitHub 호스티드
# 러너에서 빌드가 불가능했다** — 러너 루트가 72GB 에 여유 15GB 뿐이고, 가중치를 굽기도 전에
# pip 단계에서 `[Errno 28] No space left on device` 로 죽었다(실측 2026-08-16 · 빌드 #3).
# 여기서는 vLLM·torch 를 pip 휠로 직접 깔아 이미지를 ~12GB 로 낮춘다. CUDA 유저스페이스는
# torch/nvidia-* 휠이 들고 오므로 devel 이 아니라 runtime 베이스로 충분하다.
FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1

# ffmpeg — BE 가 보내는 ogg/opus·m4a 를 확실히 디코딩하려면 필요하다(server.py 주석 참고).
# python3-venv — Ubuntu 24.04 는 PEP 668 로 시스템 파이썬에 pip 설치를 막으므로 venv 를 쓴다
# (`--break-system-packages` 로 뚫는 것보다 안전하다 — OS 패키지와 섞이지 않는다).
# ★ gcc/g++ — **런타임에 필요하다.** vLLM 은 torch.compile(Inductor+Triton)로 GPU 커널을
# 그 자리에서 컴파일하므로 C 컴파일러가 없으면 엔진 기동이 죽는다(실측 2026-08-16 · 첫 배포:
# `InductorError: RuntimeError: Failed to find C compiler`). vllm-openai 베이스에는 들어
# 있었는데 CUDA runtime 베이스로 슬림화하면서 빠졌다. 수십 MB 라 이미지 크기 영향은 없다.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        python3 python3-venv python3-dev ffmpeg gcc g++ \
    && rm -rf /var/lib/apt/lists/*

ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv "$VIRTUAL_ENV"
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# ★ 가중치는 **굽지 않는다**(기본 0). RunPod 의 **Cached model** 이 같은 일을 더 잘 한다 —
# 모델을 호스트 머신에 캐시해 두고 이미 가진 호스트로 워커를 배정하며, 캐시가 없어 받는
# 동안은 과금도 하지 않는다. 마운트 위치가 `/runpod-volume/huggingface-cache/hub/` 라
# HF 규약과 같아서 HF_HOME 만 그리로 보내면 코드는 그대로 쓴다.
#
# ⚠️ 실패 이력(2026-08-16): 콜드스타트를 줄이려고 가중치를 구웠더니 이미지가 **압축 13.76GB**
# (가중치 레이어만 5.13GB)가 되어 워커가 35분 넘게 `initializing` 에서 나오지 못했다.
# 줄이려던 콜드스타트를 오히려 못 넘게 만든 셈이라 되돌린다. 굽고 싶으면 build-arg 로 1.
ARG BAKE_WEIGHTS=0
# RunPod Cached model 마운트 지점. 볼륨이 없으면 HF 가 이 경로에 직접 받는다(동작은 동일).
ENV HF_HOME=/runpod-volume/huggingface-cache
# ⚠️ 다운로드 CLI 이름이 판올림 중이다 — 최신 huggingface_hub 은 `hf`, 구버전은
# `huggingface-cli` 다. 둘 중 있는 쪽을 쓴다(한쪽만 박으면 판올림에 깨진다).
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
