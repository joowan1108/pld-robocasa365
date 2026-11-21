"""What is the residual actor outputting? Evaluate a saved specialist on states the base policy visited.

Loads runs/<model>/<task>/stage1/specialists_{latest,final}.pt and the offline cache (base-policy states with
their frozen-backbone features, proprio and VLA action a_b), and reports per action dim:
  mean / mean|.| of the deterministic residual (tanh(mu) * xi), its std across states, how often it sits at the
  bound (|a_delta| > 0.95 xi), the spread of stochastic samples, the VLA action magnitude for comparison, and the
  physical size of the residual (OSC input +-1 -> +-5 cm / +-0.5 rad per step).

    python scripts/residual_stats.py --model xr1 --task CoffeeSetupMug [--ckpt latest|final|path]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from pld_rc365.config import load_config  # noqa: E402
from pld_rc365.rl.agent import ResidualSAC  # noqa: E402
from pld_rc365.rl.buffers import ReplayBuffer  # noqa: E402

DIMS = ["dx", "dy", "dz", "rx", "ry", "rz", "grip"]
PHYS = [("cm", 5.0)] * 3 + [("rad", 0.5)] * 3 + [("", 1.0)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="xr1")
    ap.add_argument("--task", default="CoffeeSetupMug")
    ap.add_argument("--ckpt", default="latest")
    ap.add_argument("--n", type=int, default=4096)
    ap.add_argument("--samples", type=int, default=8)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    cfg = load_config(args.model, args.task, ["rl.device=cpu"])
    run = REPO / "runs" / args.model / args.task
    ck = Path(args.ckpt) if args.ckpt not in ("latest", "final") else run / "stage1" / f"specialists_{args.ckpt}.pt"
    sd = torch.load(ck, map_location="cpu", weights_only=False)
    metas = sorted((run / "offline_cache").glob("*/meta.json"))
    same = [m for m in metas if json.loads(m.read_text()).get("exec_steps") == int(cfg["base"]["exec_steps"])]
    cache = (same or metas)[0].parent
    meta = json.loads((cache / "meta.json").read_text())
    fmap, n_cams = tuple(meta["fmap_shape"]), int(meta["n_cams"])
    out = {"ckpt": str(ck), "extra": sd.get("extra", {})}
    for k, s in sd["specialists"].items():
        agent = ResidualSAC(cfg["rl"], cfg["residual"], fmap, n_cams, device="cpu", seed=0)
        agent.load_state_dict(s)
        xi = agent.xi
        buf = ReplayBuffer(200_000, (n_cams, *fmap), 16, 12, int(cfg["residual"]["dims"]),
                           views=int((meta.get("augment") or {}).get("views", 0)))
        n = buf.load(cache / f"stage{k}.npz")
        b = buf.sample(min(args.n, n), np.random.default_rng(0))
        f, p, ab = (torch.as_tensor(b[x]) for x in ("feat", "proprio", "a_b"))
        with torch.no_grad():
            det, _ = agent.actor(f, p, ab, xi, deterministic=True, with_logprob=False)
            sto = torch.stack([agent.actor(f, p, ab, xi, deterministic=False, with_logprob=False)[0]
                               for _ in range(args.samples)])
        det, sto = det.numpy(), sto.numpy()
        within = sto.std(0).mean(0)                       # stochastic spread at a fixed state
        vla = np.abs(b["a_b"][:, :7]).mean(0)
        print(f"specialist {k}: xi={xi:.3f} (xi_steps={s.get('xi_steps')}, online_steps={s.get('online_steps')}, "
              f"updates={s.get('n_updates')}) | ckpt {ck.name} {sd.get('extra', {})} | {len(det)} base-visited states")
        print(f"{'dim':>5} {'mean':>8} {'mean|.|':>8} {'|.|/xi':>7} {'std(s)':>7} {'at bound':>8} {'sample sd':>9} "
              f"{'|VLA a|':>8} {'phys mean|.|':>13}")
        rows = []
        for j, name in enumerate(DIMS):
            unit, scale = PHYS[j]
            r = {"dim": name, "mean": float(det[:, j].mean()), "mean_abs": float(np.abs(det[:, j]).mean()),
                 "frac_of_xi": float(np.abs(det[:, j]).mean() / xi), "std_states": float(det[:, j].std()),
                 "at_bound": float((np.abs(det[:, j]) > 0.95 * xi).mean()), "sample_sd": float(within[j]),
                 "vla_abs": float(vla[j]), "phys": f"{np.abs(det[:, j]).mean() * scale:.2f} {unit}".strip()}
            rows.append(r)
            print(f"{name:>5} {r['mean']:+8.3f} {r['mean_abs']:8.3f} {r['frac_of_xi']:7.2f} {r['std_states']:7.3f} "
                  f"{r['at_bound']:8.2f} {r['sample_sd']:9.3f} {r['vla_abs']:8.3f} {r['phys']:>13}")
        out[str(k)] = {"xi": xi, "rows": rows}
    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
