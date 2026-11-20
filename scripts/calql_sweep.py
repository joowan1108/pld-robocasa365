"""Calibration check of the Cal-QL critic warm-start on a saved offline buffer (no env, no VLA needed).

For sparse 0/1 rewards the pre-trained critic should satisfy Q(s, a_data) ~ Monte-Carlo return in [0, 1].
Runs Cal-QL pre-training for a few settings and reports, on a fixed held-out batch of offline transitions:
  Q(data), the MC return, their mean abs. error and correlation, Q at policy actions, Q at uniformly random
  residuals, and the TD target.

    python scripts/calql_sweep.py --model xr1 --task CoffeeSetupMug --steps 3000 \
        --variants full:5 full:1 full:0.3 full:0.1 residual:5 residual:0.3 none:0
  variant = <random-action support>:<cql alpha>   (none:0 = plain TD/SAC critic, no conservative term)
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from pld_rc365.config import load_config  # noqa: E402
from pld_rc365.rl.agent import ResidualSAC  # noqa: E402
from pld_rc365.rl.buffers import ReplayBuffer  # noqa: E402
from pld_rc365.rl.networks import combine  # noqa: E402


@torch.no_grad()
def evaluate(agent: ResidualSAC, ev: dict) -> dict:
    b = agent._to_t(ev)
    xi, d = agent.xi, agent.act_dim
    z = agent.critic.encoder(b["feat"], b["proprio"], b["a_b"])
    q_data = agent.critic(None, None, None, b["a_bar"], z=z).min(0).values
    a_pi, _ = agent.actor(b["feat"], b["proprio"], b["a_b"], xi)
    q_pi = agent.critic(None, None, None, combine(b["a_b"], a_pi, d), z=z).min(0).values
    u = (torch.rand_like(a_pi) * 2 - 1) * xi
    q_unif = agent.critic(None, None, None, combine(b["a_b"], u, d), z=z).min(0).values
    mc = b["mc_return"]
    y = agent._td_target(b, xi)
    qd, m = q_data.cpu().numpy(), mc.cpu().numpy()
    return {"q_data": float(qd.mean()), "mc": float(m.mean()), "mae": float(np.abs(qd - m).mean()),
            "corr": float(np.corrcoef(qd, m)[0, 1]), "q_pi": float(q_pi.mean()), "q_unif_residual": float(q_unif.mean()),
            "td_target": float(y.mean()), "q_min": float(qd.min()), "q_max": float(qd.max())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="xr1")
    ap.add_argument("--task", default="CoffeeSetupMug")
    ap.add_argument("--cache", default=None, help="offline_cache/<key> dir (default: the only one under the task)")
    ap.add_argument("--stage", type=int, default=0)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--variants", nargs="+", default=["full:5", "full:1", "full:0.3", "full:0.1", "none:0"])
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    cfg = load_config(args.model, args.task, args.set)
    root = Path(cfg["paths"]["runs_root"])
    root = root if root.is_absolute() else REPO / root
    if args.cache:
        cache = Path(args.cache)
    else:
        cands = sorted((root / args.model / args.task / "offline_cache").glob("*/meta.json"))
        cands = [c for c in cands if json.loads(c.read_text()).get("exec_steps") == int(cfg["base"]["exec_steps"])] or cands
        assert len(cands) == 1, f"expected exactly one offline cache, found {cands}"
        cache = cands[0].parent
    meta = json.loads((cache / "meta.json").read_text())
    fmap, n_cams = tuple(meta["fmap_shape"]), int(meta["n_cams"])
    buf = ReplayBuffer(200_000, (n_cams, *fmap), 16, 12, int(cfg["residual"]["dims"]))
    n = buf.load(cache / f"stage{args.stage}.npz")
    rng = np.random.default_rng(0)
    ev = buf.sample(min(4096, n), rng)
    print(f"cache {cache} | {n} transitions | MC return mean {buf.data['mc_return'][:n].mean():.3f} "
          f"| reward sum {buf.data['reward'][:n].sum():.0f}", flush=True)
    results = {}
    for v in args.variants:
        kind, alpha = v.split(":")
        alpha = float(alpha)
        rl = copy.deepcopy(cfg["rl"])
        rl["calql"]["form"] = "paper" if kind == "paper" else "cqlh"      # paper:<alpha> = PLD App. B.1 form
        rl["calql"]["random"] = "full" if kind in ("none", "paper") else kind
        agent = ResidualSAC(rl, cfg["residual"], fmap, n_cams, device=rl["device"], seed=0)
        t0 = time.time()
        hist = []
        for i in range(1, args.steps + 1):
            agent.update(buf.sample(agent.batch, agent.rng), calql_alpha=alpha)
            if i % args.eval_every == 0 or i == args.steps:
                r = evaluate(agent, ev)
                r["step"] = i
                hist.append(r)
                print(f"{v:>13s} step {i:6d}  Q(data) {r['q_data']:+.3f} [{r['q_min']:+.2f},{r['q_max']:+.2f}]  MC {r['mc']:.3f}  "
                      f"MAE {r['mae']:.3f}  corr {r['corr']:+.2f}  Q(pi) {r['q_pi']:+.3f}  Q(unif res) {r['q_unif_residual']:+.3f}  "
                      f"TD target {r['td_target']:+.3f}  alpha_T {agent.log_alpha.exp().item():.2f}  ({time.time() - t0:.0f}s)",
                      flush=True)
        results[v] = hist
        del agent
        torch.cuda.empty_cache()
    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
