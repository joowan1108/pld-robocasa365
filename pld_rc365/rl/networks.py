"""Residual actor / critic networks (PLD App. C.1, Table 5; SERL-style image encoder).

* Frozen ImageNet-pretrained ResNet-10 backbone (timm `resnet10t`) -> (512, 8, 8) feature map per 256x256 camera
  (4x4 at 128x128). Features are computed once at collection time and stored in the replay buffer (backbone is
  frozen). Image augmentation (random shift, pad 4) is applied before the backbone at collection time: each stored
  frame keeps the features of K shifted copies and the learner draws one per camera (residual.augment).
* Trainable SpatialLearnedEmbeddings head per camera (SERL) -> latent 256, LayerNorm, Tanh.
* Inputs besides images: proprio (16-D); the actor also gets the base action a_b (12-D): pi_delta(.|s, a_b).
  The critic is Q(s, a_bar) as in the paper's Eq. 2 (no a_b input) unless rl.critic_uses_a_b is set.
* 3-layer MLPs, hidden 256, LayerNorm + Tanh. Gaussian actor with tanh squashing, scaled to [-xi, xi].
* Clipped double Q (2 heads, shared encoder), LayerNorm.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
LOG_STD_MIN, LOG_STD_MAX = -20.0, 2.0


class FrozenBackbone(nn.Module):
    def __init__(self, name: str = "resnet10t.c3_in1k", pretrained: bool = True):
        super().__init__()
        import timm

        self.net = timm.create_model(name, pretrained=pretrained, num_classes=0, global_pool="")
        self.net.eval()
        for p in self.net.parameters():
            p.requires_grad_(False)
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

    def train(self, mode: bool = True):  # always eval (frozen BN)
        return super().train(False)

    @torch.no_grad()
    def forward(self, images_uint8: torch.Tensor, chunk: int | None = None) -> torch.Tensor:
        """images: (B, n_cam, H, W, 3) uint8 -> (B, n_cam, C, h, w) float16. `chunk`: max samples per forward pass
        (bounds the activation memory when many augmented views are encoded at once)."""
        B, n = images_uint8.shape[:2]
        outs = []
        for a in range(0, B, chunk or max(B, 1)):
            x = images_uint8[a:a + (chunk or B)].flatten(0, 1).permute(0, 3, 1, 2).float().div_(255.0)
            x = (x - self.mean) / self.std
            outs.append(self.net(x).half())
        f = torch.cat(outs, 0) if len(outs) != 1 else outs[0]
        return f.view(B, n, *f.shape[1:])


def random_shift(images_uint8: torch.Tensor, pad: int, generator: torch.Generator | None = None) -> torch.Tensor:
    """DrQ / SERL random-shift augmentation: pad every image by `pad` pixels (edge replication) and crop it back to
    its size at a random offset, i.e. shift by an integer in [-pad, pad] along each axis, drawn independently for
    every image (sample and camera). images: (B, n_cam, H, W, 3) uint8 -> same shape."""
    B, n, H, W, _ = images_uint8.shape
    dev = images_uint8.device
    sh = torch.randint(-pad, pad + 1, (B, n, 2), generator=generator).to(dev)
    rows = (torch.arange(H, device=dev)[None, None] + sh[..., 0:1]).clamp_(0, H - 1)   # (B, n, H)
    cols = (torch.arange(W, device=dev)[None, None] + sh[..., 1:2]).clamp_(0, W - 1)   # (B, n, W)
    bi = torch.arange(B, device=dev)[:, None, None, None]
    ni = torch.arange(n, device=dev)[None, :, None, None]
    return images_uint8[bi, ni, rows[:, :, :, None], cols[:, :, None, :]]


class SpatialLearnedEmbeddings(nn.Module):
    def __init__(self, c: int, h: int, w: int, num_features: int, latent: int, dropout: float = 0.0):
        super().__init__()
        self.kernel = nn.Parameter(torch.empty(c, h, w, num_features))
        nn.init.normal_(self.kernel, std=1.0 / math.sqrt(h * w))
        self.drop = nn.Dropout(dropout)
        self.proj = nn.Linear(c * num_features, latent)
        self.ln = nn.LayerNorm(latent)

    def forward(self, fmap: torch.Tensor) -> torch.Tensor:  # (B, C, h, w)
        z = torch.einsum("bchw,chwf->bcf", fmap.float(), self.kernel).flatten(1)
        return torch.tanh(self.ln(self.proj(self.drop(z))))


class Encoder(nn.Module):
    def __init__(self, n_cams: int, fmap_shape: tuple[int, int, int], proprio_dim: int, base_action_dim: int,
                 latent: int, num_spatial_features: int):
        super().__init__()
        c, h, w = fmap_shape
        self.heads = nn.ModuleList(
            [SpatialLearnedEmbeddings(c, h, w, num_spatial_features, latent) for _ in range(n_cams)]
        )
        self.proprio = nn.Sequential(nn.Linear(proprio_dim, 64), nn.LayerNorm(64), nn.Tanh())
        self.base_action_dim = int(base_action_dim)  # 0 -> a_base is ignored (paper's critic Q(s, a_bar))
        self.out_dim = n_cams * latent + 64 + self.base_action_dim

    def forward(self, feats: torch.Tensor, proprio: torch.Tensor, a_base: torch.Tensor | None) -> torch.Tensor:
        zs = [head(feats[:, i]) for i, head in enumerate(self.heads)]
        extra = [a_base] if self.base_action_dim else []
        return torch.cat(zs + [self.proprio(proprio)] + extra, dim=-1)


def mlp(inp: int, hidden: int, out: int, n_hidden: int = 3, act=nn.Tanh) -> nn.Sequential:
    layers: list[nn.Module] = []
    d = inp
    for _ in range(n_hidden):
        layers += [nn.Linear(d, hidden), nn.LayerNorm(hidden), act()]
        d = hidden
    layers.append(nn.Linear(d, out))
    return nn.Sequential(*layers)


class ResidualActor(nn.Module):
    def __init__(self, encoder: Encoder, act_dim: int, hidden: int):
        super().__init__()
        self.encoder = encoder
        self.trunk = mlp(encoder.out_dim, hidden, 2 * act_dim)
        self.act_dim = act_dim
        # per-dimension multiplier of the bound xi (residual.xi_scale); set by the agent, not part of checkpoints
        self.register_buffer("xi_scale", torch.ones(act_dim), persistent=False)

    def forward(self, feats, proprio, a_base, xi: float, deterministic: bool = False, with_logprob: bool = True,
                z: torch.Tensor | None = None):
        """Returns a_delta = xi * xi_scale * tanh(u) (|a_delta_j| <= xi * xi_scale_j) and log pi of the *normalised*
        residual tanh(u) in [-1, 1]^d.

        The density is deliberately not divided by xi^d: SAC's target entropy (-act_dim/2) assumes a unit action
        box, and with the xi schedule the maximum entropy of a_delta itself is d*log(2*xi) (-11.3 at xi=0.1), below
        the target, so the temperature diverged (first XR1 run: alpha 1 -> 54 during warm-up). Use
        `logp - d*log(xi)` where a density over a_delta is needed (Cal-QL importance weights)."""
        h = self.trunk(self.encoder(feats, proprio, a_base) if z is None else z)   # z: precomputed encoding
        mu, log_std = h.chunk(2, dim=-1)
        log_std = log_std.clamp(LOG_STD_MIN, LOG_STD_MAX)
        std = log_std.exp()
        u = mu if deterministic else mu + std * torch.randn_like(mu)
        y = torch.tanh(u)
        a_delta = xi * self.xi_scale * y
        logp = None
        if with_logprob:
            logp = (-0.5 * ((u - mu) / std) ** 2 - log_std - 0.5 * math.log(2 * math.pi)).sum(-1)
            logp = logp - (2 * (math.log(2) - u - F.softplus(-2 * u))).sum(-1)  # tanh correction
        return a_delta, logp


class Critic(nn.Module):
    def __init__(self, encoder: Encoder, act_dim: int, hidden: int, num_q: int = 2):
        super().__init__()
        self.encoder = encoder
        self.qs = nn.ModuleList([mlp(encoder.out_dim + act_dim, hidden, 1) for _ in range(num_q)])

    def forward(self, feats, proprio, a_base, a_bar, z: torch.Tensor | None = None) -> torch.Tensor:
        """Returns (num_q, B). Pass a precomputed encoding `z` to evaluate several actions cheaply."""
        if z is None:
            z = self.encoder(feats, proprio, a_base)
        x = torch.cat([z, a_bar], dim=-1)
        return torch.stack([q(x).squeeze(-1) for q in self.qs], dim=0)


def combine(a_base: torch.Tensor, a_delta: torch.Tensor, dims: int) -> torch.Tensor:
    """a_bar = clip(a_b[:dims] + a_delta, -1, 1): the full (residual-applied) arm+gripper action."""
    return (a_base[..., :dims] + a_delta).clamp(-1.0, 1.0)
