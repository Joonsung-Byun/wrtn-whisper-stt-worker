# wrtn-whisper STT worker (Qwen3-ASR-1.7B on RunPod Serverless)

OpenRouter 로 나가던 1차 텍스트 레인을 RunPod 서버리스 GPU 로 옮기기 위한 워커다.
`Qwen3-ASR-1.7B` + `Qwen3-ForcedAligner-0.6B` 를 얹고, BE 가 이미 쓰고 있는
**OpenAI 호환 `POST /audio/transcriptions`** 계약을 그대로 흉내낸다.

## 왜 이 모양인가

- **왜 Load Balancing 엔드포인트인가** — RunPod 큐 엔드포인트는 `POST /runsync` + `{"input": ...}`
  봉투를 강제해서 BE 의 전송 코드를 고쳐야 한다. Load Balancing 은 워커의 HTTP 서버로 직결돼
  커스텀 경로가 살아나므로, BE 는 **base URL 만 바뀌고 전송 코드는 그대로**다.
- **왜 1.7B 인가** — `qwen3-asr-flash` 는 오픈웨이트가 없다(API 전용). 자체 호스팅 가능한 계열은
  `0.6B` / `1.7B` / `ForcedAligner-0.6B` 뿐이다.
- **1.7B 로 내려가면 잃는 것과 얻는 것** — 본문 품질은 flash 보다 한 급 아래다(사내 실측: 같은
  300초 조각에서 글자 −13%, 중국어 혼입, 앞 절 누락). 대신 **flash 가 구조적으로 못 주던 word
  타임스탬프가 돌아온다** — flash 는 `verbose_json` 을 400 으로 거부해서 단어 하이라이팅이 꺼져
  있었다. 정렬기를 붙이면 격자 병렬·침묵 분할·단어 하이라이팅이 설정만으로 부활한다.

## API

### `POST /audio/transcriptions`

BE `server/services/whisper_hybrid.py` 의 `_whisper_chunk_request` 가 보내는 본문 그대로다.

```json
{
  "model": "Qwen/Qwen3-ASR-1.7B",
  "input_audio": { "data": "<base64>", "format": "ogg" },
  "language": "ko",
  "temperature": 0,
  "response_format": "verbose_json",
  "timestamp_granularities": ["word"]
}
```

응답 — BE 는 `timestamp_granularities` 유무에 따라 둘 중 하나만 읽는다.

```json
{ "text": "...", "usage": { "seconds": 300 },
  "words": [{ "word": "안녕하세요", "start": 0.4, "end": 1.1 }] }
```

### `GET /ping`

RunPod 헬스체크. `200` 정상 · `204` 초기화 중 · `503` 로딩 실패(사유 포함).
로딩 실패 시 프로세스를 죽이지 않는 이유는 `server.py` 주석 참고 — 죽으면 RunPod 이 재시작만
반복해서 원인이 안 보인다.

## 환경변수

| 이름 | 기본값 | 설명 |
| --- | --- | --- |
| `QWEN_ASR_MODEL` | `Qwen/Qwen3-ASR-1.7B` | 본문 모델 |
| `QWEN_ALIGNER_MODEL` | `Qwen/Qwen3-ForcedAligner-0.6B` | 타임스탬프 정렬기 |
| `QWEN_ASR_BACKEND` | `vllm` | `vllm` \| `transformers` |
| `QWEN_GPU_MEMORY_UTILIZATION` | `0.85` | vLLM VRAM 점유율 |
| `QWEN_MAX_BATCH` | `8` | 마이크로 배치 최대 건수 |
| `QWEN_BATCH_WAIT_MS` | `150` | 동료 요청 대기(0 이면 배칭 없음) |
| `STT_API_KEY` | (빈값) | 설정 시 `Authorization: Bearer` 검사 |
| `PORT` | `80` | RunPod 이 주입 |

## 배포 (RunPod)

1. 이 레포를 github.com 에 올린다(사내 엔터프라이즈 GitHub 은 Actions·RunPod 어느 쪽도
   쓸 수 없다).
2. push 하면 `.github/workflows/build.yml` 이 amd64 이미지를 구워
   `ghcr.io/<owner>/<repo>:latest` 로 올린다. 맥(arm64)에서 CUDA 이미지를 구울 필요가 없다.
3. ghcr 패키지를 **public 으로 바꾸거나**, private 로 두고 RunPod 엔드포인트에 registry
   credential 을 등록한다.
4. RunPod 콘솔 → Serverless → New endpoint → **Deploy from a Docker image** → 위 이미지.
5. **Endpoint type 을 Load balancer 로** 둔다(Queue 아님 — 위 "왜" 참고).
6. GPU 24GB 급(L4 권장 — 1.7B+0.6B 합쳐 ~8GB 라 충분하고 24GB 중 가장 싸다).
7. **네트워크 볼륨을 붙이고 `HF_HOME=/runpod-volume/hf` 를 준다.** 아래 ⚠️ 참고.
8. 배포 후 `https://<ENDPOINT_ID>.api.runpod.ai/ping` 이 200 이 되는지 먼저 확인한다.

⚠️ **가중치는 이미지에 굽지 않는다**(CI 가 `BAKE_WEIGHTS=0`). GitHub 호스티드 러너 디스크가
베이스 8GB + 가중치 8GB + 레이어 사본을 못 버틴다. 그래서 첫 부팅에 HF 에서 ~8GB 를 받는데,
**네트워크 볼륨에 `HF_HOME` 을 얹어두면 그 다운로드가 워커 간에 공유돼 한 번만 일어난다.**
볼륨 없이 띄우면 워커가 뜰 때마다 8GB 를 받고 그 시간이 그대로 과금된다.
self-hosted 러너가 있으면 `BAKE_WEIGHTS=1` 로 올려 이미지에 굽는 편이 콜드스타트에 가장 좋다.

## BE 연결

BE 는 코드 변경 없이 env 로 붙일 수 있다. **단, 지금은 1차·폴백이 같은
`OPENROUTER_BASE_URL` 을 공유한다** — 여기를 RunPod 으로 돌리면 chirp-3 폴백까지 함께 끌려간다.
폴백을 OpenRouter 에 남기려면 `_ModelSpec` 에 base_url·api_key 를 실어 1차와 폴백을 갈라야 한다
(BE 쪽 후속 작업).

폴백을 끈 채 1차만 옮기는 최소 구성:

```bash
OPENROUTER_BASE_URL=https://<ENDPOINT_ID>.api.runpod.ai
OPENROUTER_API_KEY=<STT_API_KEY 와 같은 값>
OPENROUTER_STT_MODEL=Qwen/Qwen3-ASR-1.7B
OPENROUTER_STT_WORD_TIMESTAMPS=true   # 정렬기가 붙어 있으므로 되켠다
OPENROUTER_STT_AUDIO_FORMAT=ogg
OPENROUTER_STT_CHUNK_SEC=300
OPENROUTER_STT_FALLBACK_MODEL=        # 빈 값 = chirp 폴백 비활성
```
