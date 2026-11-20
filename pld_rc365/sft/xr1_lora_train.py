"""LoRA (r=32) SFT of Xiaomi-Robotics-1-RoboCasa365 on PLD data (PLD App. C.2: LoRA rank 32).

Written from scratch: the released xr1/ trainer is full fine-tuning, belongs to a different training recipe and is
shape-incompatible with the RoboCasa365 checkpoint. The loss reproduces the HF inference path of
`modeling_mibot.MiBoTForActionGeneration.forward` with gradients:
  VLM(prompt + 3 videos) with use_cache=True -> per-layer K/V conditioning the DiT,
  DiT input [sink, state_embed(4), noisy_action(16)], causal mask, positions after the VLM's max position (no +10),
  flow matching with x_t = (1-t) * noise + t * a  (t=1 is data; inference integrates x += v dt from t=0),
  target v = a - noise, t = (1 - Beta(1.5, 1)) * 0.999  (as in xr1/mibot/models/VLA/XR1.py),
  MSE over the active action dims (action_mask: dims 0..11) and valid chunk steps, `noise_repeats` noise draws per
  VLM forward.
Actions/states are raw env units (the RC365 action stats are mean 0 / std 1 on dims 0..11; the state is not
normalised by the processor), so targets are the executed 12-D env actions zero-padded to 60.

  torchrun --nproc_per_node 8 -m pld_rc365.sft.xr1_lora_train \
     --data runs/pld_data/xr1/CoffeeSetupMug_pld200 runs/pld_data/xr1/GetToastedBread_pld200 ... \
     --out runs/xr1/sft_pld200 --max-steps 10000 --global-batch-size 64
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

from ..base.xr1_server import ROBOT_TYPE, encode_inputs, pad_and_stack
from .lerobot_reader import LeRobotEpisodes

def lora_targets(n_llm_layers: int) -> str:
    """LoRA on all LLM attention/MLP projections and all DiT projections (vision tower frozen).
    The DiT only reads each LLM layer's K/V, so the last LLM layer's q/o/MLP cannot affect the action loss:
    for that layer only k_proj / v_proj are adapted."""
    last = n_llm_layers - 1
    not_last = rf"(?!{last}\.)\d+"
    return (
        rf"vlm\.model\.language_model\.layers\.{not_last}\.(self_attn\.(q_proj|k_proj|v_proj|o_proj)|mlp\.(gate_proj|up_proj|down_proj))"
        rf"|vlm\.model\.language_model\.layers\.{last}\.self_attn\.(k_proj|v_proj)"
        r"|dit\.layers\.\d+\.(attn\.(qkv_proj|o_proj)|mlp\.(gate_proj|up_proj|down_proj))"
    )
FULLY_TRAINED = ("state_projector", "action_projector", "action_output_layer", "t_embedder", "t_projector", "sink")


class XR1Dataset(torch.utils.data.Dataset):
    def __init__(self, roots, processor, crop_ratio: float, chunk: int = 16):
        self.eps = LeRobotEpisodes(roots, chunk=chunk)
        self.processor = processor
        self.crop = crop_ratio
        self.chunk = chunk

    def __len__(self):
        return len(self.eps)

    def __getitem__(self, i):
        s = self.eps.sample(i)
        enc = encode_inputs(self.processor, s["video"], s["state"], s["lang"], self.crop)
        act = np.zeros((self.chunk, 60), dtype=np.float32)
        act[:, :12] = s["action"]
        return {**{k: v for k, v in enc.items()}, "target": torch.from_numpy(act),
                "valid": torch.from_numpy(s["valid"].astype(np.float32))}


def make_collate(pad_id: int):
    def collate(items):
        out = pad_and_stack(items, pad_id)  # same batching as the inference server
        out["target"] = torch.stack([it["target"] for it in items])
        out["valid"] = torch.stack([it["valid"] for it in items])
        return out
    return collate


def condition(m, b: dict):
    """VLM forward (with grad) -> everything the DiT needs; mirrors MiBoTForActionGeneration.forward."""
    vlm_out = m.vlm(input_ids=b["input_ids"], attention_mask=b["attention_mask"],
                    pixel_values_videos=b["pixel_values_videos"], video_grid_thw=b["video_grid_thw"],
                    use_cache=True, logits_to_keep=1)
    action_mask = b["action_mask"]
    B, A, _ = action_mask.shape
    S = b["state"].shape[1]
    q_len = A + S + 1
    dev = action_mask.device
    position_ids = (torch.arange(0, q_len, device=dev).view(1, 1, -1).repeat(3, B, 1)
                    + vlm_out.position_ids.max(dim=-1)[0][..., None] + 1)
    position_embeds = m.rotary_emb(action_mask, position_ids)
    dit_mask = torch.tril(torch.ones((B, q_len, q_len), device=dev), diagonal=0)
    cache_mask = vlm_out.attention_mask[:, None, :].expand(-1, q_len, -1)
    attn_mask = torch.cat([cache_mask, dit_mask], dim=-1)[:, None].bool()
    cache = vlm_out.past_key_values
    kv = [(cache[i][0], cache[i][1]) for i in range(len(cache))]
    return {"kv": kv, "pos": tuple(position_embeds), "attn_mask": attn_mask,
            "state_embed": m.state_projector(b["state"]), "action_mask": action_mask, "B": B}


def repeat_condition(c: dict, r: int) -> dict:
    if r == 1:
        return c
    rep = lambda x: x.repeat_interleave(r, 0)  # noqa: E731
    B = c["B"]
    return {"kv": [(rep(k), rep(v)) for k, v in c["kv"]],
            "pos": tuple(rep(p) if p.shape[0] == B else p for p in c["pos"]),
            "attn_mask": rep(c["attn_mask"]), "state_embed": rep(c["state_embed"]),
            "action_mask": rep(c["action_mask"]), "B": B * r}


def velocity(m, c: dict, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    am = c["action_mask"]
    return m.dit_forward(noisy_action=x_t.to(am.dtype), t=t.to(am.dtype).view(-1, 1, 1), action_mask=am,
                         state_embed=c["state_embed"], position_embeds=c["pos"], past_key_values=c["kv"],
                         attn_mask=c["attn_mask"])


def flow_matching_loss(m, b: dict, noise_repeats: int) -> torch.Tensor:
    """`m` is the (peft-wrapped) MiBoTForActionGeneration's underlying module."""
    c = repeat_condition(condition(m, b), noise_repeats)
    rep = lambda x: x.repeat_interleave(noise_repeats, 0)  # noqa: E731
    target, valid = rep(b["target"]), rep(b["valid"])
    n = target.shape[0]
    beta = torch.distributions.Beta(torch.tensor(1.5, device=target.device), torch.tensor(1.0, device=target.device))
    t = ((1 - beta.sample((n,))) * 0.999).to(target.dtype).view(-1, 1, 1)
    noise = torch.randn_like(target)
    x_t = (1 - t) * noise + t * target
    pred = velocity(m, c, x_t, t)
    v = target - noise
    w = c["action_mask"] * valid[..., None]
    return (((pred.float() - v.float()) ** 2) * w).sum() / w.sum().clamp_min(1.0)


class LossModule(torch.nn.Module):
    """Routes the loss through forward() so DDP's reducer sees every step."""

    def __init__(self, peft_model, noise_repeats: int):
        super().__init__()
        self.peft_model = peft_model
        self.noise_repeats = noise_repeats

    def forward(self, batch: dict) -> torch.Tensor:
        return flow_matching_loss(self.peft_model.base_model.model, batch, self.noise_repeats)


def save_merged(model, processor_src: Path, out: Path):
    merged = model.merge_and_unload()
    merged = merged.to(torch.bfloat16)
    merged.save_pretrained(out, safe_serialization=True)
    for f in processor_src.iterdir():  # custom code, tokenizer, processor/preprocessor configs, chat template
        if f.is_file() and not f.name.endswith(".safetensors") and f.name not in ("model.safetensors.index.json", "config.json"):
            shutil.copy2(f, out / f.name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model-path", default="XiaomiRobotics/Xiaomi-Robotics-1-RoboCasa365")
    ap.add_argument("--max-steps", type=int, default=10000)
    ap.add_argument("--global-batch-size", type=int, default=64)
    ap.add_argument("--per-device-batch-size", type=int, default=2,
                    help="2 fits a 24 GB RTX 4090 (no grad checkpointing: the DiT needs the VLM K/V cache)")
    ap.add_argument("--attn", default="flash_attention_2", choices=["flash_attention_2", "eager"])
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-5)
    ap.add_argument("--warmup-ratio", type=float, default=0.05)
    ap.add_argument("--lora-rank", type=int, default=32)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--noise-repeats", type=int, default=4)
    ap.add_argument("--crop-ratio", type=float, default=0.95)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--save-steps", type=int, default=0, help="0 = only final")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry-run-steps", type=int, default=0, help=">0: overfit sanity check on the first batch")
    args = ap.parse_args()

    ddp = "RANK" in os.environ
    if ddp:
        dist.init_process_group("nccl")
        rank, world = dist.get_rank(), dist.get_world_size()
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    else:
        rank, world = 0, 1
    dev = torch.device("cuda", torch.cuda.current_device())
    torch.manual_seed(args.seed + rank)
    out = Path(args.out)
    if rank == 0:
        out.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    from huggingface_hub import snapshot_download
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModel, AutoProcessor

    src = Path(args.model_path) if Path(args.model_path).is_dir() else Path(snapshot_download(args.model_path))
    processor = AutoProcessor.from_pretrained(src, trust_remote_code=True, use_fast=False)
    assert ROBOT_TYPE in processor.list_robot_types()
    model = AutoModel.from_pretrained(src, trust_remote_code=True, attn_implementation=args.attn,
                                      dtype=torch.bfloat16)
    model.requires_grad_(False)
    model = get_peft_model(model, LoraConfig(r=args.lora_rank, lora_alpha=args.lora_alpha, lora_dropout=0.0,
                                             target_modules=lora_targets(model.config.vlm_config.text_config.num_hidden_layers),
                                             bias="none"))
    inner = model.base_model.model
    for name in FULLY_TRAINED:
        getattr(inner, name).requires_grad_(True)
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.float()
    model.to(dev)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if rank == 0:
        print(f"trainable params: {n_train/1e6:.1f}M", flush=True)

    ds = XR1Dataset(args.data, processor, args.crop_ratio)
    sampler = torch.utils.data.distributed.DistributedSampler(ds, world, rank, shuffle=True, seed=args.seed) if ddp else None
    dl = torch.utils.data.DataLoader(ds, batch_size=args.per_device_batch_size, sampler=sampler,
                                     shuffle=sampler is None, num_workers=args.num_workers, drop_last=True,
                                     collate_fn=make_collate(processor.tokenizer.pad_token_id),
                                     persistent_workers=args.num_workers > 0)
    accum = max(1, args.global_batch_size // (args.per_device_batch_size * world))
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    warm = int(args.warmup_ratio * args.max_steps)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / max(1, warm) if s < warm
        else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, args.max_steps - warm))))
    loss_mod = LossModule(model, args.noise_repeats)
    if ddp:
        loss_mod = torch.nn.parallel.DistributedDataParallel(loss_mod, device_ids=[dev.index],
                                                             find_unused_parameters=True)

    log = open(out / f"train_rank{rank}.jsonl", "a") if rank == 0 else None
    step, epoch, t_loop = 0, 0, time.time()
    fixed_batch = None
    model.train()
    while step < args.max_steps:
        if sampler is not None:
            sampler.set_epoch(epoch)
        for i, batch in enumerate(dl):
            if args.dry_run_steps:
                fixed_batch = fixed_batch or batch
                batch = fixed_batch
            batch = {k: v.to(dev, non_blocking=True) for k, v in batch.items()}
            for k in ("pixel_values_videos", "state", "action_mask"):
                batch[k] = batch[k].to(torch.bfloat16)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = loss_mod(batch) / accum
            loss.backward()
            if (i + 1) % accum:
                continue
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            if log and (step % 20 == 0 or step == 1):
                log.write(json.dumps({"step": step, "loss": loss.item() * accum, "lr": sched.get_last_lr()[0],
                                      "elapsed_s": time.time() - t_start}) + "\n")
                log.flush()
            if args.save_steps and step % args.save_steps == 0 and rank == 0:
                torch.save({n: p.detach().cpu() for n, p in model.named_parameters() if p.requires_grad},
                           out / f"trainable_step{step}.pt")
            if step >= args.max_steps or (args.dry_run_steps and step >= args.dry_run_steps):
                break
        epoch += 1
        if args.dry_run_steps and step >= args.dry_run_steps:
            break
    train_s = time.time() - t_loop
    if ddp:
        dist.barrier()
    if rank == 0:
        model.eval()
        save_merged(model, src, out / "merged")
        timing = {"train_loop_seconds": train_s, "total_seconds": time.time() - t_start, "steps": step,
                  "world_size": world, "global_batch_size": args.per_device_batch_size * world * accum,
                  "num_samples": len(ds), "data": args.data, "trainable_params": n_train, "lora_rank": args.lora_rank}
        (out / "sft_timing.json").write_text(json.dumps(timing, indent=2))
    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
