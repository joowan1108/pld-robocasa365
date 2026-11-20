# Vast.ai 템플릿 (인스턴스를 잡으면 실험 1이 바로 시작되게)

이미지는 SkillPatch(DSRL) 프로젝트에서 만든 `joowan1108/dsrl-robocasa365:v1`을 그대로 씁니다.
이 저장소용으로 바뀌는 것은 **On-start Script**와 **환경변수** 두 가지뿐입니다.

## 1. 인스턴스를 잡은 뒤 자동으로 일어나는 일

| 순서 | 내용 | 시간 |
|---|---|---|
| 1 | GitHub에서 이 저장소를 clone(이미 있으면 pull) | 수 초 |
| 2 | GPU, EGL 확인과 GitHub, HF 접속 확인 | 수 초 |
| 3 | 에셋(~10GB)과 체크포인트(~24GB) 다운로드를 백그라운드로 시작. 이미 받은 건 건너뜀 | 네트워크에 따라 수십 분 |
| 4 | 이미지 환경에 우리 패키지 추가 (`scripts/vast_setup.sh`) | ~1분 |
| 5 | 에셋과 해당 모델 체크포인트를 기다림 | |
| 6 | `scripts/run_experiment1.sh` 실행 | |

6번(`run_experiment1.sh`)은 다음 순서로 진행됩니다.
1. base 평가 (fixed, pretrain, target, forgetting)
2. 성공률 확인. 기준보다 낮으면 여기서 멈춤
3. Stage 1 → Stage 2
4. 서버를 끄고 SFT
5. SFT 모델로 평가
6. `runs/summary.md` 작성

- 진행 상황은 `tail -f /workspace/autorun.log`로 봅니다. 다운로드 로그는 `/workspace/download.log`입니다.
- 끝나면 `/workspace/AUTORUN_DONE` 파일이 생기고, 실패하면 `/workspace/AUTORUN_FAILED` 파일에 이유가 남습니다.
- 인스턴스가 재시작되면 On-start가 다시 실행되는데, 이미 끝난 단계는 결과 파일이 있으니 건너뛰고 이어서 진행합니다.

## 2. 템플릿 값 (cloud.vast.ai/templates, 팀 계정)

SkillPatch 템플릿을 복제한 뒤 아래만 바꾸면 됩니다.

| 항목 | 값 |
|---|---|
| Template Name | `joowan-pld-robocasa365-v1` |
| Image Path:Tag | `joowan1108/dsrl-robocasa365:v1` |
| Docker Options | `-e NVIDIA_DRIVER_CAPABILITIES=all` |
| Environment Variables | `MUJOCO_GL=egl`, `PYOPENGL_PLATFORM=egl`, `PYTHONUNBUFFERED=1`, `MODEL=xr1`, `TASK=CoffeeSetupMug` |
| Launch Mode | SSH |
| On-start Script | 아래 4줄 |
| Docker Repository Authentication | SkillPatch와 동일 (사용자명 + Read-only 토큰, 직접 입력) |
| Disk Space | 150 GB |
| Template Visibility | Private |
| Extra Filters | `cpu_arch in ['amd64'] cuda_max_good>=12.4` |

On-start Script (코드 블록 표시 없이 이 4줄만 붙여 넣기):

```bash
cd /workspace
if [ ! -d residual ]; then git clone https://github.com/joowan1108/pld-robocasa365.git residual; fi
cd residual && git pull --ff-only || true
nohup bash scripts/autorun.sh >> /workspace/autorun.log 2>&1 &
```

저장소가 공개라 토큰이 필요 없습니다. 실험을 바꾸려면 템플릿 환경변수만 바꾸면 됩니다.

| 변수 | 기본값 | 의미 |
|---|---|---|
| `MODEL` | `xr1` | `xr1` 또는 `rldx` |
| `TASK` | `CoffeeSetupMug` | 고정 scene이 정해진 task만 가능 (나머지 둘은 `stage0_sweep` 이후) |
| `N` | `200` | PLD 성공 에피소드 수 |
| `SFT_STEPS` / `SFT_BS` / `PER_DEV` | `3000` / `32` / `2` | SFT 예산. 4090에서 OOM이 나면 `PER_DEV=1` |
| `BASE_MIN_SR` | `0.10` | base 성공률 게이트. `0`이면 끔 |
| `ATTN` | `flash_attention_2` | XR1 attention 구현. 안 되면 `eager` |

## 3. 대여 조건

- **RTX 4090 ×1 또는 ×2**. 이미지가 40 시리즈 기준이라 5090은 쓰지 않습니다.
- 2장이면 GPU마다 VLA 서버를 하나씩 띄워 Stage 1·2 처리량이 약 2배가 됩니다. 그러려면 env 8개를 돌릴 RAM 96GB 이상, vCPU 16 이상이 필요합니다.
- RAM: env 하나에 ~6.5GB, 여기에 replay buffer 최대 ~16GB가 더해집니다. env 수는 RAM을 보고 자동으로 정해집니다.
- vCPU 16 이상, 디스크 150GB, 다운로드 500Mbps 이상, Reliability 99% 이상.
- 중국 지역은 제외합니다(HF, Docker Hub 접속 문제).
- 같은 가격이면 네트워크 속도, 신뢰도, 대여 가능 기간을 우선합니다.

## 4. 끝난 뒤

1. `runs/`를 PC로 받습니다. 결과 파일이 많으면 ssh tar 스트림 여러 개를 병렬로 씁니다.
2. 원격과 로컬의 `md5sum` 목록을 비교합니다.
3. 체크포인트와 `hf_cache`는 받지 않습니다. 다시 내려받을 수 있습니다.
4. **Destroy**합니다. Stop은 디스크 요금이 계속 나갑니다.
