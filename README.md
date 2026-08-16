# wrtn-whisper STT worker (Qwen3-ASR-1.7B on RunPod Serverless)

OpenRouter 로 나가던 STT 를 RunPod 서버리스 GPU 로 옮기기 위한 워커.
**공식 vLLM 이미지에 오디오 확장만 얹은 20줄짜리 Dockerfile 이 전부다.**

## 왜 이렇게 작은가

처음에는 `qwen-asr` 패키지 + 직접 짠 FastAPI 서버로 쌓았다. 전부 불필요했다 —
**vLLM 이 Qwen3-ASR 을 정식 지원하고 OpenAI 호환 `/v1/audio/transcriptions` 를 그대로 준다.**
커스텀 이미지는 RunPod 워커가 로그 한 줄 없이 `initializing` 에서 나오지 못했고 원인을
특정하지 못한 반면, 공식 이미지는 같은 조건에서 뜨고 에러도 정확히 뱉었다. 직접 쌓지 않는다.

## 실측 (2026-08-17 · PRO 6000 MIG 24GB · US-WA-1)

| 항목 | 값 |
| --- | --- |
| 30초 오디오 전사 | **5.5초** (실시간의 1/5.5) |
| 5초 오디오 전사 | 2.6초 |
| 콜드 스타트 | **176~246초** |
| └ 이미지 pull + 컨테이너 | ~4분 (머신 캐시되면 사라짐) |
| └ **엔진 초기화** | `--enforce-eager` 로 **8분 → 대폭 감소** ★ 최대 병목이었다 |
| └ 모델 가중치 다운로드 | **17초** (1.7B 는 작다 — 여기 최적화할 가치 없음) |
| 요금 | $0.69/hr, 도는 동안만 |

⚠️ **"모델을 미리 받아두면 콜드스타트가 준다" 는 틀렸다.** 다운로드는 17초뿐이고 병목은
엔진 초기화였다. 가중치를 이미지에 굽는 시도(압축 13.76GB)는 오히려 워커를 못 뜨게 만들었다.

## 구성

```
이미지        ghcr.io/joonsung-byun/wrtn-whisper-stt-worker:<커밋 sha>
시작 명령     Qwen/Qwen3-ASR-1.7B --host 0.0.0.0 --port 80
              --gpu-memory-utilization 0.85 --max-model-len 8192 --enforce-eager
엔드포인트    Load balancer (Queue 아님) · 헬스체크 /health
컨테이너 디스크 60GB · HTTP 포트 80 · 워커 0~2 · FlashBoot on · 네트워크 볼륨 없음
```

## 밟은 지뢰 (같은 함정 반복 금지)

1. **시작 명령에 `vllm serve` 를 쓰지 않는다.** 베이스 ENTRYPOINT 가 이미 `vllm serve` 라
   중복되어 `unrecognized arguments` 로 2초 만에 죽는다. **모델명과 옵션만** 넣는다.
2. **`vllm/vllm-openai` 태그는 최신이어야 한다.** v0.14.0 은 `qwen3_asr` 아키텍처를 모른다
   ("Transformers does not recognize this architecture"). 지원은 0.27 대부터.
3. **공식 이미지에는 오디오 확장이 없다.** 그대로 쓰면 전사 요청이
   `Failed to load audio via soundfile: ImportError('Please install vllm[audio]')` 로 400.
   → 이 레포가 `librosa`·`soundfile` 을 얹는 이유.
4. **RunPod 은 `:latest` 를 캐시해 새 빌드를 자동으로 받지 않는다.** 반드시 **커밋 sha 태그**로
   Manage → New release. 같은 태그를 다시 넣으면 배포 버튼이 잠긴다.
5. **실패한 워커를 RunPod 이 자동으로 걷지 않는다.** 롤아웃 중 구 릴리스 워커가 트래픽을
   가로채 "고쳤는데 같은 에러" 처럼 보인다. Workers 탭에서 `Outdated` 를 직접 종료하고,
   **워커 버전이 전부 Latest 인지 확인한 뒤** 테스트한다. 이 함정에 세 번 속았다.

## API

```bash
curl -H "Authorization: Bearer $RUNPOD_API_KEY" \
     -F "file=@meeting.wav" -F "model=Qwen/Qwen3-ASR-1.7B" \
     https://<ENDPOINT_ID>.api.runpod.ai/v1/audio/transcriptions
# → {"text": "...", "usage": {"type": "duration", "seconds": 30}}
```

⚠️ **multipart 파일 업로드**다. BE 현행(`whisper_hybrid._whisper_chunk_request`)은
JSON + base64(`input_audio.data`) 로 보내므로 **BE 전송 코드 수정이 필요하다.**
단어 타임스탬프는 이 경로에 없다(현행 flash 와 동일하게 꺼진 상태).

## 콜드 스타트 대응

3~4시간에 한 번 쓰는 패턴이면 매 요청이 콜드 스타트다. 그러나 전사는 백그라운드 작업이고
**업로드 시작 시점에 워커를 깨우면** 업로드+화자분리(~2분) 뒤에 전사가 오므로 3분짜리
콜드 스타트가 대체로 가려진다 — 업로드→요약 전체가 5분 안에 들어오는 그림.

상시 켜기는 월 27만 원(Pod)~50만 원(서버리스 상시)이라 이 용량에는 과하다.
**3분마다 핑을 보내 살려두는 것도 결국 24시간 과금이라 같은 이야기다.**
