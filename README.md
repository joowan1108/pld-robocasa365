# PLD on RoboCasa365 — RLDX-1-FT-RC365 / Xiaomi-Robotics-1-RoboCasa365

This repo reimplements **Probe → Learn → Distill** (Xiao et al., arXiv 2511.00091) with π0 replaced by two
RoboCasa365 VLAs. It runs on **CoffeeSetupMug**, **GetToastedBread** and **SteamInMicrowave** and measures:

| | What is timed | Where |
|---|---|---|
| Stage 1 | Building a residual specialist for one task in one fixed scene (offline successes → Cal-QL → warm-up → online RL to 80/90/95% rolling SR) | `runs/<model>/<task>/stage1/timing.json` |
| Stage 2 | Collecting the hybrid-rollout SFT dataset (time to 50/100/200 successes, plus LeRobot write) | `runs/<model>/<task>/stage2/timing.json` |
| Stage 3 | LoRA r=32 SFT of the frozen VLA on PLD data (wall-clock), and success rate before vs. after on three scene axes: **fixed** (the single Stage-1/2 scene, in-distribution), **pretrain** (other training-distribution kitchens), **target** (unseen kitchens + unseen objects, OOD). Also a forgetting check on 5 untouched tasks | `runs/<model>/_sft/pld<N>/timing.json`, `runs/<model>/<task>/eval/*.json` |

`python scripts/summarize.py` turns all of these into `runs/summary.md`.

The paper releases no code. Everything here is implemented from the text: §2, §3, §4, Appendix A (Alg. 1),
B.2 and C (Table 5).

## Ambiguities and how they were resolved

| # | Question | Decision | Source |
|---|---|---|---|
| 1 | How is "random steps" of base-policy probing sampled? | `T_base ~ U{0..⌊αT⌋}` per episode, **α = 0.8**. α is configurable via `probing.alpha`; the paper's Fig. 11 plateau is 0.6. | user choice; paper §4.5 |
| 2 | What is T? | The task horizon from the robocasa registry: 600 / 3000 / 2100 steps | user, paper-literal |
| 3 | What else did the paper randomize to cover the base distribution? | Nothing explicit. The other sources are env reset randomness (object instance and placement, robot base jitter, joint noise, via a per-episode seed) and the VLA's own flow-matching noise. We keep both and never reseed per call. | paper §3.2, §4.5 |
| 4 | Is probing used during RL training too? | Yes. Probing steps set the initial state only and are **not** stored. Warm-up episodes are base-only and are stored. | paper §3.2, Table 5 |
| 5 | What goes into the online buffer? | The **full executed action** `a_b + a_δ` (clipped) together with `a_b`. Offline and warm-up data use `a_bar = a_b[:7]`. | user |
| 6 | The TD target needs the base action at s′ | The VLA action actually used at t+1 is stored as `next_a_b`. | Eq. 2 |
| 7 | How do action chunks interact with the residual? | The VLA is queried every `exec_steps` (RLDX 8, XR1 16, matching their RC365 evals). The residual acts on every env step, using the current chunk element `a_b,t`, which is also an actor/critic input. | not in paper |
| 8 | Which dimensions does the residual cover? | 7-D: eef Δpos, Δrot, gripper. Base motion, torso and mode pass through from the VLA. ξ = 0.5 in controller-normalized units. | paper 7-DoF, Table 5 |
| 9 | ξ is "tuned by a scheduler", but Table 5 gives 0.5 | **Linear ramp 0.1 → 0.5** over the first 50k *residual-acted* online steps. Warm-up and probing steps do not count. Stage 2 and eval reload the ξ the actor finished training with. The form and length are not in the paper (`residual.xi_*`). | user; paper §3.1, B.2 |
| 10 | Cal-QL steps and α | 20k steps, α = 5, MC returns from sparse rewards. The actor is randomly re-initialized afterwards. | Alg. 1; steps/α not given |
| 11 | Long composite tasks | Per-stage specialists with sparse per-stage rewards, switched by a state machine on sim-state predicates (`pld_rc365/envs/stages.py`). This follows the paper's real-world long-horizon (YAM) recipe. The toaster `wait` stage is base-only. | user; paper §4.4 |
| 12 | Residual observation | Frozen ImageNet ResNet-10 (`timm resnet10t`) on 128² `agentview_left` + `eye_in_hand`, SERL spatial embeddings, proprio (16), `a_b` (12). | Table 5 |
| 13 | Scene | One fixed (layout, style) per task with `split=None` and pretrain object instances. CoffeeSetupMug uses (11, 34); the other two come from `stage0_sweep`. | user ("scene 고정") |
| 14 | Residual during Stage-2 collection | Deterministic mean of π_δ. If the base succeeds before takeover, the episode is kept (it is a valid τ_demo) and flagged `pure_base`. | not in paper |
| 15 | PLD size | Collect 200 successes per task and write nested 50 ⊂ 100 ⊂ 200 datasets. SFT pools the 3 tasks, PLD data only, starting from each FT-RC365 checkpoint. | user |
| 16 | SFT recipe | **LoRA r=32 for both models.** RLDX uses its official trainer's LoRA flags (action model + top-4 LLM layers). XR1 uses **our own trainer** (`sft/xr1_lora_train.py`), because the released xr1/ code does full fine-tuning for a different recipe and does not fit this checkpoint. Both use lr 1e-4 cosine and the same step budget for every data size. | user; App. C.2 |

**Fallback to consider if results are poor (probing depth is uneven across tasks):** the horizon is a time limit, not
a success length, so the same α probes to very different depths per task. Base success-length medians from earlier
rollouts were ~200 / ~1540 / ~920 steps. With α=0.8, max T_base = 480 / 2400 / 1680, which is 2.3× / 1.6× / 1.8×
the success length, and P(T_base > success length) ≈ 57% / 36% / 45%. If the Stage-2 `pure_base` fraction is high,
hybrid SR collapses, or SFT gains are weak (especially on CoffeeSetupMug), consider:
* T = per-task×model mean base success length, taken from Stage-1's 50 offline successes (a `probing.T_mode` option,
  not yet implemented);
* and/or α = 0.6.

For the α ablation, plot the x-axis both as α and as T_base / success length.

**Known property of the paper-faithful setup:** with ξ = 0.5 and a gripper command near ±1, thresholded at 0.5,
the residual can **never flip the gripper**. It can only reshape arm motion. The same holds for the paper's LIBERO
setting.

## Layout

```
pld_rc365/
  envs/rc365_env.py     robocasa harness (create_env split=None + fixed layout/style), images flipped like the gym wrapper,
                        12-D action conversion, per-episode seeds, stage machine, on-disk frame recorder; VecEnv workers
  envs/stages.py        online stage predicates (coffee / toast: press, wait, pick, place / steam: 4 stages)
  base/{rldx,xr1}_server.py  ZeroMQ base-policy servers (run in each model's own venv), exact RC365 eval preprocessing
  base/client.py        history buffers [-6,-4,-2,0], chunk queue, a_b lookahead
  rl/                   ResNet-10 encoder, residual Gaussian actor, CDQ critic, SAC + Cal-QL, buffers, async learner
  rollout.py            vectorized engine: base / pld episodes, probing, storage policies, per-stage routing
  stage0_sweep.py       pick the fixed scene per task
  stage1_train.py       Stage 1;  stage2_collect.py  Stage 2;  eval.py  success rates
  data/lerobot_writer.py  RC365-format LeRobot v2.1 writer (same features/order as robocasa's converter)
  sft/run_sft.py        Stage 3 launcher + timing;  sft/xr1_lora_train.py  XR1 LoRA trainer;  sft/lerobot_reader.py
configs/                common.yaml (Table 5 etc.), models/*.yaml, tasks/*.yaml
scripts/summarize.py    tables;  scripts/run_pipeline.sh  full run for one model
tests/                  sim-free tests (fake env/base) + tiny-random-XR1 tests of the SFT forward
```

## Setup (remote GPU server)

1. **PLD venv:** Python 3.10/3.11, robocasa 1.0.1 (`pip install -e robocasa`, assets downloaded), `pip install -e .`,
   and EGL rendering (`MUJOCO_GL=egl`). This venv also needs `lerobot==0.3.3` (a robocasa dependency) for the
   Stage-2 writer. Optionally set `PLD_EGL_DEVICES=0,1` to spread env rendering across GPUs.
2. **RLDX server:** clone `RLWRLD/RLDX-1` at tag v1.0.2 and `uv sync`. Then:
   `cd RLDX-1 && uv run python /path/to/residual/pld_rc365/base/rldx_server.py --model-path RLWRLD/RLDX-1-FT-RC365 --port 21000 --device cuda:0`
   (`uv pip install pyzmq` if missing).
3. **XR1 server:** a py3.11 venv with torch 2.8, transformers 4.57.1, flash-attn 2, pyzmq, pillow, av, peft. Then:
   `MIBOT_SERVER_SEED=0 python pld_rc365/base/xr1_server.py --port 22000`
4. Put the RL learner on a different GPU from the server with `--set rl.device=cuda:1`.

## Experiment 1 — run this first

This is the smallest end-to-end run that answers the two main questions:

* **Does PLD work in the single training kitchen?** Eval axis `fixed`.
* **Does the VLA keep, or improve, its generalization when the kitchen changes every episode?** Eval axes
  `pretrain` and `target`, plus the forgetting check. This tests the paper's §4.5 claim that base-aligned PLD data
  forgets less.

Default setup: XR1 × CoffeeSetupMug (fixed scene 11/34, horizon 600), 200 PLD episodes, single-task LoRA r=32 SFT.

```bash
MODEL=xr1 BASE_PY=~/venvs/xr1/bin/python bash scripts/run_experiment1.sh
```

Results:

* SFT output goes to `runs/<model>/_sft/pld200_CoffeeSetupMug`.
* Eval tag is `sft1_pld200`.
* `summarize.py` shows it in the "All evaluations" table, next to `base`, with the number of distinct kitchens per
  axis (fixed should be 1).

Run the full grid (3 tasks pooled × 50/100/200 × 2 models), the α ablation and multi-kitchen coverage only after
this.

## Running on Vast.ai (lab image `joowan1108/dsrl-robocasa365:v1`)

Use the same template as the SkillPatch/DSRL project. See `SkillPatch/docker/VAST_TEMPLATE.md` and `vastai.md`
on the lab PC.

* **What the image provides:** the three Python envs under `/opt`.
* **What the On-start script does:** downloads the RoboCasa assets and both checkpoints into
  `/workspace/checkpoints`.
* **Which GPU to rent:** **2× RTX 4090** (1× also works). The image's torch builds target 40-series; a 5090
  would need cu128 reinstalls. With 2 GPUs `run_experiment1.sh` puts the VLA server on GPU0, env rendering + the
  residual learner on GPU1, and runs SFT with DDP on both.
* **Instance size:** ≥ 64 GB RAM, ≥ 16 vCPU, 150 GB disk, outside China. Right after renting, check:
  * `curl -I https://github.com` and `curl -I https://huggingface.co`
  * `cat /sys/fs/cgroup/memory.max`
* **Stage 3:** XR1 LoRA SFT is tight on a 24 GB card. Try `PER_DEV=1`, or run Stage 3 on a 48 GB card
  (L40S / A6000). `timing.json` records the GPU, so report it alongside the SFT time.

```bash
# put the code on the instance (GitHub clone, or from the PC:  tar --exclude=runs -czf - . | ssh -p PORT root@IP "mkdir -p /workspace/residual && tar xzf - -C /workspace/residual")
cd /workspace/residual
bash scripts/vast_setup.sh && source vast.env      # adds pyzmq/timm/peft/... to the baked envs, writes vast.env
bash scripts/run_experiment1.sh                    # unattended, resumable; starts/stops the VLA server itself
```

Operating rules (learned on the DSRL runs):

* Never run another GPU job (e.g. an eval) on a GPU that is training. Simulator rendering alone has
  OOM-killed full runs.
* Stop processes by PID, as `scripts/server.sh` does, not with `pkill -f`.
* Never overwrite a bash script while it is running.
* When finished, download `runs/` and compare md5 lists between remote and local before **Destroy**. Stop keeps
  billing for disk. Don't download checkpoints or `hf_cache`; they can be re-downloaded.
* Never put tokens on the instance or in git.

## Running on 1–2× RTX 4090 / 5090 (24 / 32 GB)

| | 1 GPU | 2 GPUs |
|---|---|---|
| Base-policy server | `CUDA_VISIBLE_DEVICES=0` | `CUDA_VISIBLE_DEVICES=0` |
| PLD process (env rendering + residual learner) | `--set rl.device=cuda:0`, `PLD_EGL_DEVICES=0` | `--set rl.device=cuda:1`, `PLD_EGL_DEVICES=1` |
| SFT (stop the server first) | `NPROC=1` | `NPROC=2` (DDP, full model copy per GPU) |

Approximate GPU memory:

| Component | Memory |
|---|---|
| XR1 bf16 inference | ~11 GB |
| RLDX bf16 inference | ~14 GB |
| Residual learner + frozen ResNet-10 | ~1–2 GB |
| MuJoCo EGL rendering, 4 envs × 3 cams (default `env.num_envs=4`) | a few GB |
| Host RAM per RoboCasa env | ~6.5 GB |

On the DSRL runs, XR1 + learner + 4 envs used 20–23 GB of a 4090. RLDX is ~3 GB larger; use `env.num_envs=3` if it OOMs.

SFT notes:

* `--per-device-batch-size 2` (24 GB) or `4` (32 GB); the rest of the effective batch comes from gradient
  accumulation.
* XR1 cannot use gradient checkpointing, because the DiT consumes the VLM's K/V cache.
* Experiment 1 defaults: effective batch 32, 3000 steps.

5090 (Blackwell) notes:

* The RLDX venv already uses torch 2.7 + flash-attn 2.7.4, and the RLDX repo benchmarks on 5090.
* For XR1, if flash-attn won't build, use `--attn eager` on the server and `ATTN=eager` for SFT. It is slower and
  uses more memory. MiBoT has no sdpa path.

Host RAM: the online buffer is ~64 KB per transition, so ~16 GB at the paper's 250k capacity. On a 32 GB machine,
use `--set rl.buffer_capacity=100000`.

The Stage 1/2 bottleneck is VLA inference: one server, with B=1 calls for XR1. Every `timing.json` records the GPU
model and count, so times measured on a 4090 and a 5090 stay distinguishable.

## Run order

```bash
python -m pld_rc365.stage0_sweep --model xr1 --task GetToastedBread --episodes 10   # repeat for rldx and SteamInMicrowave
#   -> write layout_id / style_id into configs/tasks/*.yaml (same scene for both models, base SR ~15-70%)
python -m pld_rc365.eval --model xr1 --task CoffeeSetupMug --tag base          # fixed + pretrain + target
python -m pld_rc365.stage1_train   --model xr1 --task CoffeeSetupMug
python -m pld_rc365.stage2_collect --model xr1 --task CoffeeSetupMug
python -m pld_rc365.sft.run_sft --model xr1 --n 200 --python ~/venvs/xr1/bin/python --nproc 8
#   restart the server with --model-path runs/xr1/_sft/pld200/merged   (RLDX: runs/rldx/_sft/pld200/ckpt/checkpoint-*)
python -m pld_rc365.eval --model xr1 --task CoffeeSetupMug --tag sft_pld200
python scripts/summarize.py
```

Override any config value with `--set a.b=c`, e.g. `--set env.num_envs=3 rl.buffer_capacity=100000 probing.alpha=0.6`.

Memory: each specialist's online buffer stores fp16 ResNet feature maps for obs and next obs, about 64 KB per
transition. At the paper's 250k capacity that is about 16 GB of host RAM per specialist. The arrays are allocated
lazily, so memory grows with what is actually stored.

## Verification

* **Sim-free tests:** `pytest tests`. These cover the runner, buffers, SAC/Cal-QL, stage machine, and Stage-1/2
  orchestration end to end with fakes.
* **Tiny XR1 tests:** `XR1_HF_DIR=<dir with the HF repo's .py/.json/tokenizer files> pytest tests/test_xr1_tiny.py`.
  These build a tiny random MiBoT from the real `modeling_mibot.py` and check that:
  * the SFT conditioning/velocity path equals the model's inference forward **bitwise**;
  * a right-padded batch matches per-sample results;
  * LoRA and projector gradients flow;
  * the merged checkpoint reloads.
* **On the server, before Stage 1:**
  1. Base SR on random pretrain scenes should be roughly in line with the published per-task numbers
     (XR1: CoffeeSetupMug 25/50, GetToastedBread 18/50, SteamInMicrowave 32/50 on its seeds). This catches
     image, state or action-format bugs.
  2. Run `stage1_train --set residual.xi=0.0 rl.warmup_episodes=0 ...` briefly. The online SR should equal the base SR.
  3. The stage predicates should fire in order on the offline successes. Check `stage` in `stage1/episodes.jsonl`.
* **Before Stage 3:**
  1. Load one written dataset with the RLDX loader, and with `pld_rc365.sft.lerobot_reader`. Compare its frame
     orientation with an official RC365 LeRobot episode.
  2. Run `xr1_lora_train --dry-run-steps 50` on one task. The loss should fall (overfit check).
  3. The LoRA checkpoint at step 0 (`--max-steps 1`) should reproduce the base SR.

## Not verified here

This development machine has no GPU and no robocasa install. The env harness, both servers, the RLDX trainer
invocation and the LeRobot writer follow the upstream source line by line but have **not been executed**. Run the
server-side checks above first.
