"""Turn residual_demo recordings into videos that show when the residual takes over and what it outputs.

  python scripts/render_demo.py --hybrid <ep dir> [--vla <ep dir>] --out clip.mp4 [--title "..."]
  python scripts/render_demo.py --pair-root runs/xr1/CoffeeSetupMug/demo --vla-tag vla --hybrid-tags xi01,xi02 --out-dir videos

Every row is one episode: the three cameras, a status badge (VLA only / VLA + residual), a timeline of the whole
episode with the takeover step marked and the residual (dx, dy, dz) drawn along it, and bars with the residual and the
VLA action at the current step. With --vla the top row is the VLA alone on the same seed.
Pair mode: for every takeover episode that succeeded, the VLA-only episode of the same seed that failed (if any)."""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

import av
import numpy as np
from PIL import Image, ImageDraw, ImageFont

W, IMG, HEAD, PANEL = 768, 256, 30, 150
ROW = HEAD + IMG + PANEL
COL = {"dx": (230, 80, 70), "dy": (60, 170, 90), "dz": (70, 130, 230)}
ORANGE, GREY, BG, FG, DIM = (240, 140, 30), (120, 120, 120), (18, 18, 20), (235, 235, 235), (150, 150, 155)
GREEN, RED = (70, 190, 100), (225, 70, 70)


def font(size: int, bold: bool = False):
    for p in (f"/usr/share/fonts/truetype/dejavu/DejaVuSans{'-Bold' if bold else ''}.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(p, size)
        except OSError:
            continue
    return ImageFont.load_default()


F_S, F_M, F_B = font(13), font(15), font(17, True)


class Episode:
    def __init__(self, d: str | Path, horizon: int):
        d = Path(d)
        self.frames = np.load(d / "frames.npy", mmap_mode="r")
        low = np.load(d / "lowdim.npz", allow_pickle=True)
        self.n = int(low["length"])
        self.seed = int(low["seed"])
        self.meta = ast.literal_eval(str(low["meta"]))
        tr = np.load(d / "trace.npy")
        self.a_b, self.a_d, self.acted, self.decided = tr[:, 1:8], tr[:, 8:15], tr[:, 15] > 0, tr[:, 16] > 0
        self.takeover = self.meta.get("takeover_step")
        self.horizon = horizon
        self.success = self.n < horizon
        bound = np.abs(self.a_d[:, :3]).max() if self.acted.any() else 0.0
        self.scale = max(float(bound), 0.05)

    def timeline(self, xi: float | None) -> Image.Image:
        """Static background of the timeline strip: axis, takeover marker, residual traces of the whole episode."""
        h = 62
        im = Image.new("RGB", (W, h), BG)
        dr = ImageDraw.Draw(im)
        x0, x1, mid = 46, W - 14, h // 2 + 2
        X = lambda t: x0 + (x1 - x0) * t / self.horizon   # noqa: E731
        dr.rectangle([X(0), mid - 22, X(self.n), mid + 22], fill=(34, 34, 38))
        if self.takeover is not None:
            dr.rectangle([X(self.takeover), mid - 22, X(self.n), mid + 22], fill=(58, 40, 22))
        dr.line([X(0), mid, X(self.horizon), mid], fill=(70, 70, 75))
        for t in range(0, self.horizon + 1, 100):
            dr.line([X(t), mid + 22, X(t), mid + 27], fill=DIM)
        s = self.scale
        for j, k in enumerate(("dx", "dy", "dz")):
            pts = [(X(t), mid - 20 * float(np.clip(self.a_d[t, j] / s, -1, 1))) for t in range(self.n)]
            if len(pts) > 1 and self.acted.any():
                dr.line(pts, fill=COL[k], width=1)
        if self.takeover is not None:
            dr.line([X(self.takeover), mid - 26, X(self.takeover), mid + 26], fill=ORANGE, width=2)
        dr.text((4, mid - 28), f"+{s:.2f}", font=F_S, fill=DIM)
        dr.text((4, mid + 14), f"-{s:.2f}", font=F_S, fill=DIM)
        self._tl = (x0, x1, mid)
        return im


def bar(dr, x, y, w, val, scale, color, label, ref=None):
    """Horizontal signed bar in [-scale, scale]; `ref` (the VLA action) is a thin grey bar underneath."""
    c = x + w // 2
    dr.rectangle([x, y, x + w, y + 12], outline=(60, 60, 65))
    dr.line([c, y - 2, c, y + 14], fill=DIM)
    v = float(np.clip(val / scale, -1, 1))
    dr.rectangle([min(c, c + v * w / 2), y + 1, max(c, c + v * w / 2), y + 11], fill=color)
    if ref is not None:
        r = float(np.clip(ref / scale, -1, 1))
        dr.rectangle([min(c, c + r * w / 2), y + 15, max(c, c + r * w / 2), y + 18], fill=GREY)
    dr.text((x - 26, y - 3), label, font=F_M, fill=color)
    dr.text((x + w + 8, y - 3), f"{val:+.3f}", font=F_M, fill=FG)
    if ref is not None:
        dr.text((x + w + 78, y - 3), f"VLA {ref:+.3f}", font=F_S, fill=DIM)


def render_row(ep: Episode, t: int, title: str, xi: float | None, tl: Image.Image) -> Image.Image:
    im = Image.new("RGB", (W, ROW), BG)
    dr = ImageDraw.Draw(im)
    f = min(t, ep.n)                      # frame index: observation before step t (the last one after the end)
    for j in range(3):
        im.paste(Image.fromarray(np.asarray(ep.frames[f, j])), (j * IMG, HEAD))
    step = min(t, ep.n - 1)
    on = bool(ep.acted[step]) and t < ep.n
    ended = t >= ep.n
    # header
    dr.text((8, 5), title, font=F_B, fill=FG)
    if ended:
        txt, col = (f"SUCCESS at step {ep.n}", GREEN) if ep.success else (f"FAILED (time limit {ep.horizon})", RED)
    elif on:
        txt, col = "VLA + RESIDUAL", ORANGE
    else:
        txt, col = "VLA only", GREY
    tw = dr.textlength(txt, font=F_B)
    dr.rectangle([W - tw - 22, 3, W - 4, HEAD - 3], fill=col)
    dr.text((W - tw - 13, 5), txt, font=F_B, fill=(10, 10, 10))
    if on or ended:
        c = col if ended else ORANGE
        dr.rectangle([0, HEAD, W - 1, HEAD + IMG - 1], outline=c, width=3)
    # timeline
    y0 = HEAD + IMG + 4
    im.paste(tl, (0, y0))
    x0, x1, mid = ep._tl
    xc = x0 + (x1 - x0) * min(t, ep.n) / ep.horizon
    dr.line([xc, y0 + mid - 28, xc, y0 + mid + 28], fill=FG, width=2)
    if ep.takeover is not None:
        xt = x0 + (x1 - x0) * ep.takeover / ep.horizon
        dr.text((min(xt + 4, W - 150), y0 - 1), f"takeover at step {ep.takeover}", font=F_S, fill=ORANGE)
    # readout
    y1 = y0 + 66
    dr.text((8, y1 + 2), f"step {min(t, ep.n):3d}", font=F_B, fill=FG)
    dr.text((8, y1 + 26), f"seed {ep.seed}", font=F_S, fill=DIM)
    if ep.takeover is None:
        dr.text((8, y1 + 44), "no residual", font=F_S, fill=DIM)
    else:
        dr.text((8, y1 + 44), f"xi {xi:.2f}" if xi else "", font=F_S, fill=DIM)
    s = ep.scale
    for j, k in enumerate(("dx", "dy", "dz")):
        bar(dr, 150, y1 + 4 + j * 24, 300, float(ep.a_d[step, j]) if on else 0.0, s, COL[k], k, ref=float(ep.a_b[step, j]))
    if on and ep.decided[step]:
        dr.ellipse([124 - 14, y1 + 60, 124 - 4, y1 + 70], fill=ORANGE)
    dr.text((628, y1 + 4), "colour: residual", font=F_S, fill=DIM)
    dr.text((628, y1 + 20), "grey: VLA action", font=F_S, fill=DIM)
    dr.text((628, y1 + 36), "(end-effector move,", font=F_S, fill=DIM)
    dr.text((628, y1 + 52), "same scale, clipped)", font=F_S, fill=DIM)
    return im


def render(hybrid: str, vla: str | None, out: str, horizon: int, xi: float | None, title: str = "", fps: int = 20,
           tail: int = 40):
    rows = []
    if vla:
        e = Episode(vla, horizon)
        e.scale = xi or e.scale            # same scale as the takeover row
        rows.append((e, f"VLA alone  -  {title}".strip(" -"), None, e.timeline(None)))
    e = Episode(hybrid, horizon)
    e.scale = xi or e.scale
    rows.append((e, f"VLA -> residual takeover  -  {title}".strip(" -"), xi, e.timeline(xi)))
    T = max(r[0].n for r in rows) + tail
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    box = av.open(out, "w")
    st = box.add_stream("libx264", rate=fps)
    st.width, st.height, st.pix_fmt = W, ROW * len(rows), "yuv420p"
    st.options = {"crf": "21", "preset": "medium"}
    for t in range(T):
        frame = np.concatenate([np.asarray(render_row(ep, t, ti, x, tl)) for ep, ti, x, tl in rows], 0)
        for pkt in st.encode(av.VideoFrame.from_ndarray(np.ascontiguousarray(frame), format="rgb24")):
            box.mux(pkt)
    for pkt in st.encode():
        box.mux(pkt)
    box.close()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hybrid")
    ap.add_argument("--vla")
    ap.add_argument("--out")
    ap.add_argument("--title", default="")
    ap.add_argument("--xi", type=float, default=None)
    ap.add_argument("--horizon", type=int, default=600)
    ap.add_argument("--pair-root")
    ap.add_argument("--vla-tag", default="vla")
    ap.add_argument("--hybrid-tags", default="")
    ap.add_argument("--out-dir")
    ap.add_argument("--max-per-tag", type=int, default=6)
    ap.add_argument("--include-failures", type=int, default=0, help="per tag: also render this many failed takeovers")
    a = ap.parse_args()
    if a.hybrid:
        print(render(a.hybrid, a.vla, a.out, a.horizon, a.xi, a.title))
        return
    root = Path(a.pair_root)
    ld = lambda p: [json.loads(l) for l in open(p)] if Path(p).exists() else []   # noqa: E731
    vla = ld(root / a.vla_tag / "results.jsonl")
    index = []
    for tag in [t for t in a.hybrid_tags.split(",") if t]:
        res = ld(root / tag / "results.jsonl")
        # most convincing first: seeds where the VLA alone failed every time
        def vla_fail_rate(seed):
            v = [r for r in vla if r["seed"] == seed]
            return (sum(not r["success"] for r in v) / len(v)) if v else -1.0
        wins = sorted([r for r in res if r["success"] and r["took_over"]],
                      key=lambda r: (-vla_fail_rate(r["seed"]), r["seed"], r["t_base"]))
        fails = [r for r in res if not r["success"] and r["took_over"]][: a.include_failures]
        seen = set()
        for r in wins[: a.max_per_tag * 3] + fails:
            key = (r["seed"], r["t_base"], r["success"])
            if key in seen or (r["success"] and sum(1 for i in index if i["tag"] == tag and i["success"]) >= a.max_per_tag):
                continue
            seen.add(key)
            vf = [v for v in vla if v["seed"] == r["seed"] and not v["success"]]
            vs = [v for v in vla if v["seed"] == r["seed"]]
            pick = (vf or vs or [None])[0]
            name = f"{tag}_seed{r['seed']}_t{r['t_base']}_{'success' if r['success'] else 'fail'}.mp4"
            title = f"seed {r['seed']}, {tag}"
            render(r["rec_dir"], pick["rec_dir"] if pick else None, str(Path(a.out_dir) / name), a.horizon, r["xi"], title)
            index.append({"file": name, "tag": tag, "seed": r["seed"], "t_base": r["t_base"], "success": r["success"],
                          "length": r["length"], "xi": r["xi"], "hold": r["hold"],
                          "vla_alone_same_seed": {"episodes": len(vs), "failures": len(vs) - sum(v["success"] for v in vs)},
                          "vla_row_success": bool(pick["success"]) if pick else None})
            print("rendered", name, flush=True)
    (Path(a.out_dir) / "index.json").write_text(json.dumps(index, indent=2))


if __name__ == "__main__":
    main()
