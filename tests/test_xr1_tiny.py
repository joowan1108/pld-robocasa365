"""Checks the XR1 SFT forward against the model's own inference forward, using the real
modeling_mibot.py / processing_mibot.py with a tiny random config (CPU, eager attention).

Needs a local copy of the HF repo files (no weights): set XR1_HF_DIR to a dir with config.json,
configuration_mibot.py, modeling_mibot.py, processing_mibot.py, tokenizer + processor json files.
"""

import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

HF = os.environ.get("XR1_HF_DIR")
pytestmark = pytest.mark.skipif(not HF, reason="XR1_HF_DIR not set")


def _tiny_dir(tmp: Path) -> Path:
    src = Path(HF)
    d = tmp / "tiny_xr1"
    d.mkdir()
    for f in src.iterdir():
        if f.is_file() and f.suffix in (".py", ".json", ".txt", ".jinja") and f.name != "model.safetensors.index.json":
            shutil.copy2(f, d / f.name)
    cfg = json.loads((src / "config.json").read_text())
    tc, vc = cfg["vlm_config"]["text_config"], cfg["vlm_config"]["vision_config"]
    tc.update(hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=4,
              num_key_value_heads=2, head_dim=16)
    tc["rope_scaling"]["mrope_section"] = [4, 2, 2]
    vc.update(depth=2, hidden_size=32, intermediate_size=64, num_heads=2, out_hidden_size=64,
              deepstack_visual_indexes=[1])
    cfg["dit_config"].update(hidden_size=64, num_hidden_layers=2, num_key_value_heads=2, head_dim=16)
    (d / "config.json").write_text(json.dumps(cfg))
    return d


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    import torch
    from transformers import AutoConfig, AutoModel, AutoProcessor

    d = _tiny_dir(tmp_path_factory.mktemp("xr1"))
    config = AutoConfig.from_pretrained(d, trust_remote_code=True)
    config.vlm_config._attn_implementation = "eager"
    torch.manual_seed(0)
    model = AutoModel.from_config(config, trust_remote_code=True, attn_implementation="eager").to(torch.bfloat16).eval()
    proc = AutoProcessor.from_pretrained(d, trust_remote_code=True, use_fast=False)
    return model, proc


def _sample(rng, lang):
    from pld_rc365.base.xr1_server import CAMS

    video = {c: rng.integers(0, 255, (4, 256, 256, 3), dtype=np.uint8) for c in CAMS}
    state = {"end_effector_position_relative": rng.normal(size=(4, 3)).astype(np.float32),
             "end_effector_rotation_relative": rng.normal(size=(4, 4)).astype(np.float32),
             "gripper_qpos": rng.normal(size=(4, 2)).astype(np.float32),
             "base_position": rng.normal(size=(4, 3)).astype(np.float32),
             "base_rotation": rng.normal(size=(4, 4)).astype(np.float32)}
    return video, state, lang


def test_condition_velocity_matches_inference_forward(tiny):
    import torch

    from pld_rc365.base.xr1_server import encode_inputs
    from pld_rc365.sft.xr1_lora_train import condition, velocity

    model, proc = tiny
    rng = np.random.default_rng(0)
    enc = dict(encode_inputs(proc, *_sample(rng, "pick the mug"), 0.95))
    enc = {k: (v.to(torch.bfloat16) if torch.is_floating_point(v) else v) for k, v in enc.items()}
    torch.manual_seed(123)
    ref = model(**enc, num_steps=1).actions                     # x1 = x0 + v(x0, t=0)
    torch.manual_seed(123)
    x0 = torch.randn_like(enc["action_mask"])
    with torch.no_grad():
        c = condition(model, enc)
        ours = x0 + velocity(model, c, x0, torch.zeros(1))
    torch.testing.assert_close(ours.float(), ref.float(), rtol=0, atol=0)  # same ops, same dtype -> bitwise


def test_right_padded_batch_equals_individual(tiny):
    import torch

    from pld_rc365.base.xr1_server import encode_inputs
    from pld_rc365.sft.xr1_lora_train import condition, make_collate, velocity

    model, proc = tiny
    rng = np.random.default_rng(1)
    items = []
    for lang in ("pick the mug", "pick the yellow and white striped coffee mug from the counter"):
        enc = dict(encode_inputs(proc, *_sample(rng, lang), 0.95))
        enc["target"] = torch.zeros(16, 60)
        enc["valid"] = torch.ones(16)
        items.append(enc)
    assert items[0]["input_ids"].shape[1] != items[1]["input_ids"].shape[1]
    batch = make_collate(proc.tokenizer.pad_token_id)(items)
    batch = {k: (v.to(torch.bfloat16) if torch.is_floating_point(v) else v) for k, v in batch.items()}
    x = torch.randn(2, 16, 60, dtype=torch.bfloat16)
    t = torch.tensor([0.3, 0.7])
    with torch.no_grad():
        vb = velocity(model, condition(model, batch), x, t)
        for j, it in enumerate(items):
            single = {k: (v.to(torch.bfloat16) if torch.is_floating_point(v) else v) for k, v in it.items()
                      if k not in ("target", "valid")}
            vs = velocity(model, condition(model, single), x[j:j + 1], t[j:j + 1])
            torch.testing.assert_close(vb[j:j + 1].float(), vs.float(), rtol=3e-2, atol=3e-2)


def test_loss_backward_reaches_lora_and_projectors(tiny):
    import torch
    from peft import LoraConfig, get_peft_model

    from pld_rc365.base.xr1_server import encode_inputs
    from pld_rc365.sft.xr1_lora_train import FULLY_TRAINED, LossModule, lora_targets, make_collate

    model, proc = tiny
    import copy
    m = copy.deepcopy(model).train()
    m.requires_grad_(False)
    n_layers = m.config.vlm_config.text_config.num_hidden_layers
    m = get_peft_model(m, LoraConfig(r=4, lora_alpha=4, target_modules=lora_targets(n_layers)))
    for n in FULLY_TRAINED:
        getattr(m.base_model.model, n).requires_grad_(True)
    names = [n for n, p in m.named_parameters() if p.requires_grad]
    assert any("language_model" in n and "lora_A" in n for n in names)
    assert any("dit.layers" in n and "lora_A" in n for n in names)
    assert not any("visual" in n for n in names)
    rng = np.random.default_rng(2)
    enc = dict(encode_inputs(proc, *_sample(rng, "pick the mug"), 0.95))
    enc["target"] = torch.randn(16, 60) * 0.1
    enc["valid"] = torch.ones(16)
    b = make_collate(proc.tokenizer.pad_token_id)([enc])
    b = {k: (v.to(torch.bfloat16) if torch.is_floating_point(v) else v) for k, v in b.items()}
    loss = LossModule(m, noise_repeats=2)(b)
    loss.backward()
    assert torch.isfinite(loss)
    grads = {n: p.grad for n, p in m.named_parameters() if p.requires_grad}
    # The DiT reads every LLM layer's K/V only, so the last layer's q/o/MLP cannot influence the loss;
    # lora_targets() therefore leaves them out. Everything else trainable must get a gradient.
    assert all(g is not None for g in grads.values()), [n for n, g in grads.items() if g is None][:5]


def test_merged_checkpoint_reloads_with_same_outputs(tiny, tmp_path):
    import copy

    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModel

    from pld_rc365.base.xr1_server import encode_inputs
    from pld_rc365.sft.xr1_lora_train import lora_targets, save_merged

    model, proc = tiny
    m = copy.deepcopy(model)
    m = get_peft_model(m, LoraConfig(r=4, lora_alpha=4, target_modules=lora_targets(2), init_lora_weights=False))
    src = Path(proc.tokenizer.name_or_path)
    save_merged(m, src, tmp_path / "merged")
    re = AutoModel.from_pretrained(tmp_path / "merged", trust_remote_code=True, attn_implementation="eager",
                                   dtype=torch.bfloat16).eval()
    enc = dict(encode_inputs(proc, *_sample(np.random.default_rng(3), "pick the mug"), 0.95))
    enc = {k: (v.to(torch.bfloat16) if torch.is_floating_point(v) else v) for k, v in enc.items()}
    merged_in_mem = copy.deepcopy(m).merge_and_unload().to(torch.bfloat16).eval()
    torch.manual_seed(0)
    a = merged_in_mem(**enc, num_steps=2).actions
    torch.manual_seed(0)
    b = re(**enc, num_steps=2).actions
    torch.testing.assert_close(a.float(), b.float(), rtol=1e-2, atol=1e-3)  # bf16 round-trip
    assert (tmp_path / "merged" / "processing_mibot.py").exists()
    assert (tmp_path / "merged" / "tokenizer.json").exists()


def test_batched_inference_forward_equals_per_env(tiny):
    """The server's batched handler: model.forward on a right-padded batch == per-env B=1 calls with the same noise."""
    import torch

    from pld_rc365.base.xr1_server import encode_inputs, pad_and_stack

    model, proc = tiny
    rng = np.random.default_rng(5)
    encs = []
    for lang in ("pick the mug", "pick the yellow and white striped coffee mug from the counter", "open the door"):
        e = dict(encode_inputs(proc, *_sample(rng, lang), 0.95))
        encs.append({k: (v.to(torch.bfloat16) if torch.is_floating_point(v) else v) for k, v in e.items()})
    batch = pad_and_stack(encs, proc.tokenizer.pad_token_id)
    noise = torch.randn(len(encs), 16, 60).to(torch.bfloat16)
    real = torch.randn_like
    try:  # inject the same initial noise in both paths
        torch.randn_like = lambda x, **kw: noise[: x.shape[0]].clone() if x.shape[0] == len(encs) else None
        with torch.no_grad():
            out_b = model(**batch, num_steps=3).actions
        outs = []
        for j, e in enumerate(encs):
            torch.randn_like = lambda x, j=j, **kw: noise[j:j + 1].clone()
            with torch.no_grad():
                outs.append(model(**e, num_steps=3).actions)
    finally:
        torch.randn_like = real
    for j, o in enumerate(outs):  # every env, incl. the right-padded shorter prompts
        torch.testing.assert_close(out_b[j:j + 1].float(), o.float(), rtol=3e-2, atol=3e-2)


def test_full_finetune_grads_and_reload(tiny, tmp_path):
    """--mode full: input embeddings frozen, vision tower / LLM / DiT trained with gradient checkpointing; the loss
    with checkpointing equals the loss without; the saved model reloads with the same outputs."""
    import copy

    import torch
    from transformers import AutoModel

    from pld_rc365.base.xr1_server import encode_inputs
    from pld_rc365.sft.xr1_lora_train import (LossModule, full_ft_param_groups, make_collate, save_merged,
                                              setup_full_finetune)

    model, proc = tiny
    m = copy.deepcopy(model).train()
    ref = copy.deepcopy(model).train()
    setup_full_finetune(m)
    ref.requires_grad_(True)
    ref.vlm.get_input_embeddings().requires_grad_(False)
    for mod in (m, ref):          # as in main(): trainable weights fp32, frozen ones bf16, bf16 autocast
        for p in mod.parameters():
            if p.requires_grad:
                p.data = p.data.float()
    names = {n for n, p in m.named_parameters() if p.requires_grad}
    emb = m.vlm.get_input_embeddings().weight
    assert not emb.requires_grad
    assert any("visual" in n for n in names) and any("dit.layers" in n for n in names)
    groups = full_ft_param_groups(m, 0.1)
    assert sum(len(g["params"]) for g in groups) == len(names) and groups[1]["weight_decay"] == 0.0
    rng = np.random.default_rng(7)
    enc = dict(encode_inputs(proc, *_sample(rng, "pick the mug"), 0.95))
    enc["target"] = torch.randn(16, 60) * 0.1
    enc["valid"] = torch.ones(16)
    b = make_collate(proc.tokenizer.pad_token_id)([enc])
    for k in ("pixel_values_videos", "state", "action_mask"):
        b[k] = b[k].to(torch.bfloat16)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        torch.manual_seed(0)
        loss = LossModule(m, noise_repeats=2)(b)
        torch.manual_seed(0)
        loss_ref = LossModule(ref, noise_repeats=2)(b)
    torch.testing.assert_close(loss, loss_ref)          # checkpointing does not change the forward
    loss.backward()
    got = {n for n, p in m.named_parameters() if p.requires_grad and p.grad is not None and p.grad.abs().sum() > 0}
    assert any("visual" in n for n in got), "vision tower gets no gradient"
    assert any("dit.layers" in n for n in got) and any("language_model.layers.0." in n for n in got)
    # save (bf16) and reload: same actions as the in-memory bf16 model
    m.eval()
    save_merged(m, Path(proc.tokenizer.name_or_path), tmp_path / "full")
    re = AutoModel.from_pretrained(tmp_path / "full", trust_remote_code=True, attn_implementation="eager",
                                   dtype=torch.bfloat16).eval()
    e2 = dict(encode_inputs(proc, *_sample(np.random.default_rng(8), "pick the mug"), 0.95))
    e2 = {k: (v.to(torch.bfloat16) if torch.is_floating_point(v) else v) for k, v in e2.items()}
    mm = copy.deepcopy(m).to(torch.bfloat16).eval()
    with torch.no_grad():
        torch.manual_seed(0)
        a = mm(**e2, num_steps=2).actions
        torch.manual_seed(0)
        c = re(**e2, num_steps=2).actions
    torch.testing.assert_close(a.float(), c.float(), rtol=1e-2, atol=1e-3)
