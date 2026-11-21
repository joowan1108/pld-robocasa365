"""Stage 1 -- residual RL specialist(s) for one task in one fixed scene (PLD Alg. 1).

Phases (each timed in runs/<model>/<task>/stage1/timing.json):
  offline_collect : base-policy rollouts until n=50 successes -> offline buffers (per stage segment, MC returns)
  calql_pretrain  : Cal-QL critic warm-start per specialist; actor re-initialised afterwards
  warmup          : 100 base-only episodes stored in the online buffers (learners already running)
  online          : PLD episodes: T_base ~ U[0, alpha*T] probing (not stored) then residual takeover (stored)
                    until rolling SR (window 100, takeover episodes) >= stop_sr, or budget exhausted
  final_eval      : clean evaluation of pi_bar (no probing, residual mean) on the fixed eval seed list

Usage:
  python -m pld_rc365.stage1_train --model xr1 --task CoffeeSetupMug [--set env.num_envs=16 ...]
"""

from __future__ import annotations

import collections
import hashlib
import itertools
import json
import time
from pathlib import Path

import numpy as np
import torch

from .common import (aug_views, build_backbone, build_env_and_base, build_specialists, common_argparser, learnable_stage_ids,
                     make_runner, sample_t_base, save_specialists, sync_collectors)
from .config import REPO_ROOT, load_config, run_dir
from .rl.agent import AsyncLearner
from .rollout import EpisodeResult, EpisodeSpec
from .utils.timing import JsonlLogger, Timer


def init_seed_pool(cfg: dict) -> list[int]:
    """stage1.init_seed_pool = N > 0: every Stage-1 episode (offline, warm-up, online, final eval) resets from one of
    N fixed seeds, i.e. a small fixed set of initial states (like LIBERO's fixed init states) instead of a new object
    instance / placement every episode. Diagnostic for whether residual RL can learn in the simplest setting."""
    n = int(cfg["stage1"].get("init_seed_pool", 0) or 0)
    return [int(cfg["seeds"]["offline"]) + i for i in range(n)]


def seed_stream(cfg: dict, base: int):
    pool = init_seed_pool(cfg)
    return itertools.cycle(pool) if pool else itertools.count(base)


def offline_cache_dir(cfg: dict, spec, n_off: int) -> Path:
    root = Path(cfg["paths"]["runs_root"])
    root = root if root.is_absolute() else REPO_ROOT / root
    # exec_steps (how many actions of each VLA chunk are executed before re-planning) changes the base policy's
    # behaviour, so it is part of the key: runs with a different value never overwrite each other's buffers.
    key = f"L{spec.layout_id}_S{spec.style_id}_n{n_off}_e{int(cfg['base']['exec_steps'])}"
    if init_seed_pool(cfg):
        key += f"_p{len(init_seed_pool(cfg))}"
    if int(cfg["residual"]["image_size"]) != 128:   # buffers hold encoder features of the residual images; 128 was
        key += f"_i{int(cfg['residual']['image_size'])}"   # the only size before 2026-11-21 (keys without a suffix)
    if len(cfg["residual"]["cameras"]) != 2:        # likewise 2 cameras and no augmented views
        key += f"_c{len(cfg['residual']['cameras'])}"
    if aug_views(cfg):
        key += f"_v{aug_views(cfg)}p{int(cfg['residual']['augment'].get('pad', 4))}"
    hold, nst = int(cfg.get("rl", {}).get("residual_hold", 1)), int(cfg.get("rl", {}).get("n_step", 1))
    if hold > 1 or nst > 1:                          # stored transitions are block / n-step aggregates
        key += f"_h{hold}n{nst}"
    return root / cfg["model_name"] / cfg["task_name"] / "offline_cache" / key


def offline_meta(cfg: dict, spec, fmap_shape, n_cams: int, n_off: int) -> dict:
    """Everything the stored transitions depend on; a reuse must match all of it."""
    return {
        "model": cfg["model_name"], "task": cfg["task_name"], "layout_id": spec.layout_id, "style_id": spec.style_id,
        "obj_instance_split": cfg["env"].get("obj_instance_split"), "stage_names": list(spec.stage_names),
        "residual_cameras": list(cfg["residual"]["cameras"]), "image_size": int(cfg["residual"]["image_size"]),
        "fmap_shape": list(fmap_shape), "n_cams": n_cams, "gamma": float(cfg["rl"]["discount"]),
        "n_successes": n_off, "base_model_path": cfg["base"]["model_path"], "exec_steps": int(cfg["base"]["exec_steps"]),
        "offline_seed_base": int(cfg["seeds"]["offline"]),
        **({"init_seed_pool": len(init_seed_pool(cfg))} if init_seed_pool(cfg) else {}),
        **({"augment": dict(cfg["residual"]["augment"])} if aug_views(cfg) else {}),
        # only when not 1/1, so buffers written before these options existed still match
        **({"residual_hold": int(cfg["rl"].get("residual_hold", 1)), "n_step": int(cfg["rl"].get("n_step", 1))}
           if int(cfg["rl"].get("residual_hold", 1)) > 1 or int(cfg["rl"].get("n_step", 1)) > 1 else {}),
    }


def save_offline_cache(cache: Path, specs: dict, meta: dict) -> None:
    import subprocess

    cache.mkdir(parents=True, exist_ok=True)
    for k, sp in specs.items():
        sp.offline.save(cache / f"stage{k}.npz")
    try:
        meta["git_commit"] = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT,
                                            capture_output=True, text=True).stdout.strip()
    except Exception:  # noqa: BLE001
        pass
    (cache / "meta.json").write_text(json.dumps(meta, indent=2, default=float))
    print(f"[stage1] offline buffer saved to {cache}", flush=True)


# Settings that do not change what Cal-QL + warm-up produce (or only marginally: xi enters the TD target through
# pi_delta's action at s'). They are recorded in the warm-up cache but not part of its key, so e.g. a xi change
# can reuse the critic and warm-up data.
WARMUP_KEY_EXCLUDE_RL = {"device", "prefetch_threads", "prefetch_depth"}
WARMUP_KEY_EXCLUDE_RES = {"xi", "xi_schedule", "xi_start", "xi_ramp_steps"}


def warmup_meta(cfg: dict, offline_src: dict) -> dict:
    """Everything the post-warm-up state (Cal-QL critic, warm-up transitions, learner state) depends on."""
    return {
        "offline_cache_created": offline_src.get("created"),  # a re-collected offline buffer invalidates it
        "offline_key": {k: offline_src.get(k) for k in ("model", "task", "layout_id", "style_id", "n_successes",
                                                         "base_model_path", "fmap_shape", "n_cams")},
        # critic_action is left out at its default so warm-up caches written before the option existed still match
        "rl": {k: v for k, v in cfg["rl"].items()
               if k not in WARMUP_KEY_EXCLUDE_RL and not (k == "critic_action" and v == "total")},
        "residual": {k: v for k, v in cfg["residual"].items() if k not in WARMUP_KEY_EXCLUDE_RES},
        "warmup_seed_base": int(cfg["seeds"]["warmup"]),
        **({"init_seed_pool": len(init_seed_pool(cfg))} if init_seed_pool(cfg) else {}),
        "stage_ids": sorted(int(k) for k in learnable_stage_ids(cfg)),
    }


def warmup_cache_dir(offline_cache: Path, meta: dict) -> Path:
    h = hashlib.sha1(json.dumps(meta, sort_keys=True, default=str).encode()).hexdigest()[:10]
    return offline_cache / f"warmup_{h}"


def save_warmup_cache(d: Path, specs: dict, meta: dict, records: list[dict], info: dict) -> None:
    """Learners must be stopped. Saves per specialist: full agent state + optimisers, and the online buffer."""
    d.mkdir(parents=True, exist_ok=True)
    for k, sp in specs.items():
        a = sp.agent
        torch.save({"agent": a.state_dict(), "critic_opt": a.critic_opt.state_dict(),
                    "actor_opt": a.actor_opt.state_dict(), "alpha_opt": a.alpha_opt.state_dict()},
                   d / f"agent{k}.pt")
        sp.online.save(d / f"online{k}.npz")
    with open(d / "warmup_episodes.jsonl", "w") as f:
        for r in records:
            f.write(json.dumps(r, default=float) + "\n")
    (d / "meta.json").write_text(json.dumps({**meta, **info}, indent=2, default=float))
    print(f"[stage1] warm-up state saved to {d}", flush=True)


def load_warmup_cache(d: Path, specs: dict, actor_mode: str) -> None:
    for k, sp in specs.items():
        a = sp.agent
        ck = torch.load(d / f"agent{k}.pt", map_location=a.device, weights_only=False)
        a.load_state_dict(ck["agent"])
        a.critic_opt.load_state_dict(ck["critic_opt"])
        if actor_mode == "keep":
            a.actor_opt.load_state_dict(ck["actor_opt"])
            a.alpha_opt.load_state_dict(ck["alpha_opt"])
        else:  # "reset": fresh pi_delta + temperature (Alg. 1 initialisation) on top of the warmed-up critic
            a.init_actor()
        a.xi_steps = 0
        sp.online.load(d / f"online{k}.npz")


def main():
    ap = common_argparser("PLD stage 1: residual specialist training")
    ap.add_argument("--skip-final-eval", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.model, args.task, args.set)
    rd = run_dir(cfg, "stage1")
    (rd / "config.json").write_text(json.dumps(cfg, indent=2))
    timer = Timer(rd / "timing.json", extra={"model": args.model, "task": args.task,
                                              "num_envs": int(cfg["env"]["num_envs"]),
                                              "base_replicas": int(cfg["base"].get("replicas", 1))})
    ep_log = JsonlLogger(rd / "episodes.jsonl")
    learn_log = JsonlLogger(rd / "learner.jsonl")
    s1, rl = cfg["stage1"], cfg["rl"]

    venv, base, spec = build_env_and_base(cfg, training=True)
    timer.set("num_envs_actual", venv.num_envs)
    horizon = venv.horizon
    K = len(spec.stage_names)
    timer.set("horizon", horizon)
    timer.set("scene", {"layout": spec.layout_id, "style": spec.style_id})
    timer.set("probing_alpha", cfg["probing"]["alpha"])
    backbone, fmap_shape, n_cams = build_backbone(cfg)
    stage_ids = learnable_stage_ids(cfg)
    specs = build_specialists(cfg, fmap_shape, n_cams, stage_ids)
    for sp in specs.values():
        sp.agent.reserve_gpu_memory()
    runner = make_runner(cfg, venv, base, backbone, specs, K)
    rng = np.random.default_rng(cfg["seeds"]["online"])

    def episode_record(phase: str, r: EpisodeResult, **kw) -> dict:
        return dict(phase=phase, seed=r.spec.seed, success=r.success, length=r.length, stage=r.stage_reached,
                    t_base=r.spec.t_base, took_over=r.took_over, takeover_step=r.takeover_step,
                    n_stored=r.n_stored, wall_s=r.wall_s, scene=r.extra.get("scene"), init=r.extra.get("init"),
                    residual=r.extra.get("residual"), **kw)

    def log_episode(phase: str, r: EpisodeResult, **kw) -> dict:
        rec = episode_record(phase, r, **kw)
        ep_log.log(**rec)
        return rec

    try:
        # ------------------------------------------------------------ offline successes
        n_off = int(rl["offline_successes"])
        seeds = seed_stream(cfg, cfg["seeds"]["offline"])
        if init_seed_pool(cfg):
            timer.set("init_seed_pool", init_seed_pool(cfg))
        counts = {"episodes": 0, "successes": 0, "succ_len": []}
        runner.offline_quota = n_off

        def off_end(r: EpisodeResult) -> bool:
            counts["episodes"] += 1
            if r.success:
                counts["successes"] += 1
                counts["succ_len"].append(r.length)
            log_episode("offline", r)
            return counts["successes"] >= n_off

        cache = offline_cache_dir(cfg, spec, n_off)
        meta_now = offline_meta(cfg, spec, fmap_shape, n_cams, n_off)
        if s1.get("reuse_offline") and (cache / "meta.json").exists():
            src = json.loads((cache / "meta.json").read_text())
            mismatch = {k: (src.get(k), v) for k, v in meta_now.items() if src.get(k) != v}
            if mismatch:
                raise RuntimeError(f"offline cache {cache} is incompatible: {mismatch}")
            with timer.phase("offline_load"):
                for k, sp in specs.items():
                    sp.offline.load(cache / f"stage{k}.npz")
            timer.set("offline", {**src["stats"], "reused_from": str(cache),
                                  "offline_transitions": {k: len(sp.offline) for k, sp in specs.items()}})
            timer.add_phase_seconds("offline_collect", float(src["stats"]["collect_seconds"]))
            timer.set("offline_collect_is_from_cache", True)
        else:
            with timer.phase("offline_collect"):
                runner.run(lambda: EpisodeSpec(seed=next(seeds), mode="base", store="offline"), off_end)
            stats = {
                "episodes": counts["episodes"], "successes": counts["successes"],
                "base_sr_fixed_scene": counts["successes"] / max(1, counts["episodes"]),
                "mean_success_length": float(np.mean(counts["succ_len"])) if counts["succ_len"] else None,
                "collect_seconds": timer.data["phases"]["offline_collect"]["seconds"],
                "num_envs": venv.num_envs,
            }
            timer.set("offline", {**stats, "offline_transitions": {k: len(sp.offline) for k, sp in specs.items()}})
            save_offline_cache(cache, specs, {**meta_now, "stats": stats, "source_run": str(rd),
                                              "hardware": timer.data["hardware"], "created": time.time()})
            timer.set("offline_cache", str(cache))

        # Warm-up cache: the state after Cal-QL + warm-up (critic, warm-up transitions, learner state) is saved under
        # the offline cache, keyed by everything it depends on except xi; stage1.reuse_warmup loads it.
        off_src = json.loads((cache / "meta.json").read_text())
        wmeta = warmup_meta(cfg, off_src)
        wdir = warmup_cache_dir(cache, wmeta)
        reuse_warm = bool(s1.get("reuse_warmup")) and (wdir / "meta.json").exists()
        if s1.get("reuse_warmup") and not reuse_warm:
            print(f"[stage1] no compatible warm-up cache ({wdir.name}); running Cal-QL + warm-up", flush=True)

        def start_learners():
            for k, sp in specs.items():
                sp.learner = AsyncLearner(sp.agent, sp.offline, sp.online, sp.collector_actor, sp.actor_lock,
                                          log_fn=lambda k=k, **kw: learn_log.log(phase="online", stage=k, **kw))
                sp.learner.start()

        def check():
            for sp in specs.values():
                if sp.learner is not None and sp.learner.error is not None:
                    raise RuntimeError("learner thread died") from sp.learner.error

        if reuse_warm:
            wsrc = json.loads((wdir / "meta.json").read_text())
            actor_mode = str(s1.get("warmup_cache_actor", "keep"))
            with timer.phase("warmup_load"):
                load_warmup_cache(wdir, specs, actor_mode)
            for line in (wdir / "warmup_episodes.jsonl").read_text().splitlines():
                ep_log.log(**{**json.loads(line), "from_cache": True})
            timer.add_phase_seconds("calql_pretrain", float(wsrc["calql_seconds"]))
            timer.add_phase_seconds("warmup", float(wsrc["warmup_seconds"]))
            timer.set("warmup_is_from_cache", {"dir": str(wdir), "actor": actor_mode,
                                               "source_run": wsrc.get("source_run"),
                                               "xi_at_creation": wsrc.get("xi_at_creation")})
            sync_collectors(specs)
            start_learners()
        else:
            # -------------------------------------------------------- Cal-QL critic warm-start
            with timer.phase("calql_pretrain"):
                for k, sp in specs.items():
                    sp.agent.pretrain_calql(sp.offline, int(rl["calql"]["steps"]),
                                            log_fn=lambda **kw: learn_log.log(phase="calql", stage=k, **kw))
            sync_collectors(specs)
            start_learners()

            # -------------------------------------------------------- warm-up (base-only, stored)
            n_warm = int(rl["warmup_episodes"])
            wseeds = seed_stream(cfg, cfg["seeds"]["warmup"])
            wcount = {"n": 0, "issued": 0}
            wrecords: list[dict] = []

            def warm_spec():
                if wcount["issued"] >= n_warm:
                    return None
                wcount["issued"] += 1
                return EpisodeSpec(seed=next(wseeds), mode="base", store="online")

            def warm_end(r):
                wcount["n"] += 1
                wrecords.append(log_episode("warmup", r))
                return False

            with timer.phase("warmup"):
                runner.run(warm_spec, warm_end, check)

            # Save the post-warm-up state (learners paused so the snapshot is consistent), then resume.
            for sp in specs.values():
                sp.learner.stop()
            save_warmup_cache(wdir, specs, wmeta, wrecords, {
                "calql_seconds": timer.data["phases"]["calql_pretrain"]["seconds"],
                "warmup_seconds": timer.data["phases"]["warmup"]["seconds"],
                "xi_at_creation": {k: sp.agent.xi for k, sp in specs.items()},
                "residual_full": cfg["residual"], "num_envs": venv.num_envs, "source_run": str(rd),
                "created": time.time()})
            timer.set("warmup_cache", str(wdir))
            sync_collectors(specs)
            start_learners()

        # ------------------------------------------------------------ online PLD training
        oseeds = seed_stream(cfg, cfg["seeds"]["online"])
        window = collections.deque(maxlen=int(s1["sr_window"]))
        window_all = collections.deque(maxlen=int(s1["sr_window"]))
        thresholds = sorted(float(x) for x in s1["sr_thresholds"])
        crossed: dict[float, float] = {}
        t_online0 = time.time()
        # Limits can be changed while the run is going: write {"max_online_env_steps": N, "stop_sr": x,
        # "max_wall_hours": h} (any subset) to <run dir>/control.json; it is re-read at every episode end.
        limits = {"max_online_env_steps": int(s1["max_online_env_steps"]), "stop_sr": float(s1["stop_sr"]),
                  "max_wall_hours": float(s1["max_wall_hours"])}
        control = rd / "control.json"
        state = {"episodes": 0, "stop_reason": None}

        def refresh_limits():
            if not control.exists():
                return
            try:
                new = {k: type(limits[k])(v) for k, v in json.loads(control.read_text()).items() if k in limits}
            except (ValueError, TypeError) as e:
                print(f"[stage1] ignoring unreadable {control}: {e}", flush=True)
                return
            changed = {k: v for k, v in new.items() if limits[k] != v}
            if changed:
                limits.update(changed)
                timer.set("limits_override", dict(limits))
                print(f"[stage1] limits changed via control.json: {changed}", flush=True)

        def online_spec():
            return EpisodeSpec(seed=next(oseeds), mode="pld", t_base=sample_t_base(cfg, horizon, rng), store="online")

        def online_end(r: EpisodeResult) -> bool:
            state["episodes"] += 1
            window_all.append(float(r.success))
            if r.took_over:
                window.append(float(r.success))
            sr = float(np.mean(window)) if window else 0.0
            log_episode("online", r, rolling_sr=sr, rolling_sr_all=float(np.mean(window_all)),
                        online_steps={k: sp.agent.online_steps for k, sp in specs.items()},
                        n_updates={k: sp.agent.n_updates for k, sp in specs.items()})
            if len(window) >= window.maxlen:
                for th in thresholds:
                    if th not in crossed and sr >= th:
                        crossed[th] = time.time()
                        timer.mark(f"sr>={th}_since_stage1_start")
                        timer.mark(f"sr>={th}_since_online_start", since=t_online0)
            if state["episodes"] % int(s1["checkpoint_every_episodes"]) == 0:
                save_specialists(specs, rd / "specialists_latest.pt", {"episodes": state["episodes"], "rolling_sr": sr})
            refresh_limits()
            if len(window) >= window.maxlen and sr >= limits["stop_sr"]:
                state["stop_reason"] = f"rolling_sr>={limits['stop_sr']}"
            elif all(sp.agent.online_steps >= limits["max_online_env_steps"] for sp in specs.values()):
                state["stop_reason"] = "max_online_env_steps"
            elif time.time() - t_online0 > limits["max_wall_hours"] * 3600:
                state["stop_reason"] = "max_wall_hours"
            return state["stop_reason"] is not None

        with timer.phase("online"):
            runner.run(online_spec, online_end, check)
        for sp in specs.values():
            sp.learner.stop()
        sync_collectors(specs)
        save_specialists(specs, rd / "specialists_final.pt", {"episodes": state["episodes"]})
        timer.set("online", {
            "episodes": state["episodes"], "stop_reason": state["stop_reason"],
            "final_rolling_sr": float(np.mean(window)) if window else None,
            "online_steps": {k: sp.agent.online_steps for k, sp in specs.items()},
            "n_updates": {k: sp.agent.n_updates for k, sp in specs.items()},
            "thresholds_not_reached": [th for th in thresholds if th not in crossed],
            "runner_stats": dict(runner.stats),
            "base_queries": base.n_queries, "base_query_seconds": base.query_seconds,
        })
        tot = sum(timer.data["phases"][p]["seconds"] for p in ("offline_collect", "calql_pretrain", "warmup", "online"))
        timer.set("stage1_total_seconds", tot)

        # ------------------------------------------------------------ clean evaluation of pi_bar
        if not args.skip_final_eval:
            n_eval = int(s1["final_eval_episodes"])
            runner.aug_views = 0   # evaluation stores nothing: no augmented views to encode
            pool = init_seed_pool(cfg)
            eseeds = iter([pool[j % len(pool)] for j in range(n_eval)] if pool else
                          range(cfg["eval"]["seed_base"], cfg["eval"]["seed_base"] + n_eval))
            res: list[EpisodeResult] = []

            def eval_spec():
                s = next(eseeds, None)
                return None if s is None else EpisodeSpec(seed=s, mode="pld", t_base=0, deterministic=True)

            def eval_end(r: EpisodeResult) -> bool:
                res.append(r)
                log_episode("final_eval", r)
                return False

            with timer.phase("final_eval"):
                runner.run(eval_spec, eval_end)
            timer.set("final_eval", {"episodes": len(res), "success_rate": float(np.mean([r.success for r in res]))})
    finally:
        for sp in specs.values():
            if sp.learner is not None and sp.learner.is_alive():
                sp.learner.stop()
        venv.close()
        ep_log.close()
        learn_log.close()


if __name__ == "__main__":
    main()
