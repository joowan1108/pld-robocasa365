"""Residual SAC specialist with Cal-QL critic warm-start (PLD §3.1, Alg. 1, Table 5).

Alg. 1 order: (1) Cal-QL pre-train the critic on the offline buffer (successful base rollouts),
(2) *randomly* (re-)initialise pi_delta, (3) online SAC with symmetric offline/online sampling,
no BC term on the actor, TD target bootstraps the combined policy a_bar = a_b + a_delta (Eq. 2).
"""

from __future__ import annotations

import copy
import math
import threading
import time

import numpy as np
import torch
import torch.nn.functional as F

from .buffers import BatchPrefetcher, ReplayBuffer, symmetric_sample
from .networks import Critic, Encoder, ResidualActor, combine


class XiSchedule:
    def __init__(self, cfg: dict):
        self.kind = cfg.get("xi_schedule", "constant")
        self.xi = float(cfg["xi"])
        self.start = float(cfg.get("xi_start", self.xi))
        self.ramp = int(cfg.get("xi_ramp_steps", 1))

    def __call__(self, online_steps: int) -> float:
        if self.kind == "constant":
            return self.xi
        frac = min(1.0, online_steps / max(1, self.ramp))
        return self.start + frac * (self.xi - self.start)


class ResidualSAC:
    def __init__(self, rl: dict, residual: dict, fmap_shape: tuple, n_cams: int, proprio_dim: int = 16,
                 base_dim: int = 12, device: str = "cuda:0", seed: int = 0):
        self.rl, self.res = rl, residual
        self.device = torch.device(device)
        self.act_dim = int(residual["dims"])
        self.gamma = float(rl["discount"])
        self.tau = float(rl["tau"])
        self.batch = int(rl["batch_size"])
        self.target_entropy = -self.act_dim * float(rl["target_entropy_scale"])
        self.xi_sched = XiSchedule(residual)
        self.online_steps = 0  # stored online transitions (budget)
        self.xi_steps = 0      # stored online transitions where the residual acted (drives the xi schedule;
                               # warm-up / base-only steps do not advance it)
        # Paper Eq. 2: critic Q(s, a_bar) -- a_b goes to the actor only. critic_uses_a_b=true gives Q(s, a_b, a_bar)
        # (used by the first runs, before this was checked against the paper).
        # rl.critic_action = residual gives Q(s, a_b, a_delta): the VLA action is part of the critic's state and the
        # action input is the applied residual alone (scaled to [-1, 1]). Deviates from Eq. 2; see `_qa`.
        self.critic_action = str(rl.get("critic_action", "total"))
        if self.critic_action not in ("total", "residual"):
            raise ValueError(f"rl.critic_action must be total|residual, got {self.critic_action!r}")
        # residual.xi_scale: per-dimension multiplier of xi (e.g. smaller bound on the rotation dims, whose VLA
        # actions are ~4x smaller than the position ones). |a_delta_j| <= xi * xi_scale_j.
        sc = residual.get("xi_scale")
        self.xi_scale_np = np.ones(self.act_dim, np.float32) if sc is None else np.asarray(sc, np.float32)
        if self.xi_scale_np.shape != (self.act_dim,) or (self.xi_scale_np <= 0).any():
            raise ValueError(f"residual.xi_scale must be {self.act_dim} positive numbers, got {sc!r}")
        self.xi_scale = torch.as_tensor(self.xi_scale_np, device=self.device)
        self.res_scale = (float(residual["xi"]) * self.xi_scale).clamp_min(1e-8)   # critic_action=residual input scale
        critic_base = base_dim if (bool(rl.get("critic_uses_a_b", False)) or self.critic_action == "residual") else 0
        self._arch = dict(fmap_shape=fmap_shape, n_cams=n_cams, proprio_dim=proprio_dim, base_dim=base_dim,
                          critic_base_dim=critic_base, critic_action=self.critic_action)
        torch.manual_seed(seed)
        self.rng = np.random.default_rng(seed)

        self.critic = Critic(self._make_encoder(critic_base), self.act_dim, rl["hidden_dim"], rl["num_q"]).to(self.device)
        self.critic_targ = copy.deepcopy(self.critic).requires_grad_(False)
        self.critic_opt = torch.optim.AdamW(self.critic.parameters(), lr=rl["lr"])
        self.init_actor()
        self.n_updates = 0

    # ------------------------------------------------------------------ construction
    def _make_encoder(self, base_dim: int | None = None) -> Encoder:
        a = self._arch
        return Encoder(a["n_cams"], a["fmap_shape"], a["proprio_dim"], a["base_dim"] if base_dim is None else base_dim,
                       self.rl["latent_dim"], self.rl["num_spatial_features"])

    def init_actor(self):
        """Random (re-)initialisation of pi_delta and the temperature (Alg. 1: after critic pre-training)."""
        self.actor = ResidualActor(self._make_encoder(), self.act_dim, self.rl["hidden_dim"]).to(self.device)
        self.actor.xi_scale.copy_(self.xi_scale)
        self.actor_opt = torch.optim.AdamW(self.actor.parameters(), lr=self.rl["lr"])
        self.log_alpha = torch.tensor(math.log(self.rl["init_temperature"]), device=self.device, requires_grad=True)
        self.alpha_opt = torch.optim.AdamW([self.log_alpha], lr=self.rl["lr"])

    @property
    def xi(self) -> float:
        return self.xi_sched(self.xi_steps)

    # ------------------------------------------------------------------ helpers
    def _to_t(self, b: dict) -> dict[str, torch.Tensor]:
        return {k: torch.as_tensor(v, device=self.device) for k, v in b.items()}

    def _qa(self, a_b: torch.Tensor, a_bar: torch.Tensor) -> torch.Tensor:
        """Action input of the critic for the executed arm action a_bar taken on top of the base action a_b.

        total    : a_bar itself -- Q(s, a_bar), paper Eq. 2.
        residual : (a_bar - clip(a_b)) / xi, the residual that was actually applied, in [-1, 1]; a_b enters through
                   the critic's encoder, so the critic is Q(s, a_b, a_delta). Base-only data (offline buffer, warm-up)
                   has exactly 0 here, so the action input cannot carry information about what the VLA is doing."""
        if self.critic_action == "total":
            return a_bar
        return (a_bar - a_b[..., : self.act_dim].clamp(-1.0, 1.0)) / self.res_scale

    def _step(self, opt, loss, params):
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, self.rl["grad_clip"])
        opt.step()

    def _td_target(self, b, xi):
        with torch.no_grad():
            nf, npp, nab = b["next_feat"], b["next_proprio"], b["next_a_b"]
            n = int(self.rl.get("otf_samples", 1))
            if n <= 1:
                a_d, logp = self.actor(nf, npp, nab, xi)
                q_next = self.critic_targ(nf, npp, nab, self._qa(nab, combine(nab, a_d, self.act_dim))).min(0).values
            else:  # on-the-fly: sample n residuals at the fixed base action, back up the max
                B = nf.shape[0]
                rep = lambda x: x.repeat_interleave(n, 0)  # noqa: E731
                a_d, logp = self.actor(rep(nf), rep(npp), rep(nab), xi)
                q = self.critic_targ(rep(nf), rep(npp), rep(nab), self._qa(rep(nab), combine(rep(nab), a_d, self.act_dim))).min(0).values
                q, arg = q.view(B, n).max(1)
                logp = logp.view(B, n).gather(1, arg[:, None]).squeeze(1)
                q_next = q
            if self.rl.get("backup_entropy", False):
                q_next = q_next - self.log_alpha.exp() * logp
            r = b["reward"] + float(self.rl.get("reward_bias", 0.0))
            return r + self.gamma * (1.0 - b["done"]) * q_next

    def _soft_update(self):
        with torch.no_grad():
            for p, pt in zip(self.critic.parameters(), self.critic_targ.parameters()):
                pt.mul_(1 - self.tau).add_(p, alpha=self.tau)

    # ------------------------------------------------------------------ losses
    def critic_loss(self, b, xi, calql_alpha: float = 0.0):
        y = self._td_target(b, xi)
        z = self.critic.encoder(b["feat"], b["proprio"], b["a_b"])
        q = self.critic(None, None, None, self._qa(b["a_b"], b["a_bar"]), z=z)  # (num_q, B)
        loss = sum(F.mse_loss(qi, y) for qi in q)
        info = {"q": q.mean().item(), "td_target": y.mean().item()}
        if calql_alpha > 0:
            if self.rl["calql"].get("form", "paper") == "paper":
                cql, cinfo = self._calql_term_paper(b, z, q, xi)
                loss = 0.5 * loss + calql_alpha * cql      # App. B.1: alpha * (...) + 1/2 * TD^2
            else:
                cql, cinfo = self._calql_term(b, z, q, xi)
                loss = loss + calql_alpha * cql
            info.update(cinfo)
        return loss, info

    def _calql_term_paper(self, b, z, q_data, xi):
        """Cal-QL exactly as written in the PLD paper (App. B.1):

            E_{s~D, a~pi}[ max(Q(s, a), V^mu(s)) ] - E_{(s,a)~D}[ Q(s, a) ]

        pi is the combined policy (a_b + pi_delta(s, a_b)); V^mu(s) is the Monte-Carlo return of the offline
        (base-policy) trajectory at s. Only policy actions are pushed down, and never below V^mu -- no uniform random
        actions and no log-sum-exp (those come from the Cal-QL code base's CQL(H) variant, see _calql_term)."""
        n = int(self.rl["calql"]["num_samples"])
        rep = lambda x: x.repeat_interleave(n, 0)  # noqa: E731
        B = z.shape[0]
        zr, ab = rep(z), rep(b["a_b"])
        with torch.no_grad():
            # encode each state once and repeat the encoding (not the feature maps) for the n action samples
            za = self.actor.encoder(b["feat"], b["proprio"], b["a_b"])
            a_pi, _ = self.actor(None, None, None, xi, z=rep(za))
        q_pi = self.critic(None, None, None, self._qa(ab, combine(ab, a_pi, self.act_dim)), z=zr)   # (num_q, B*n)
        mc = rep(b["mc_return"])[None]
        info = {"q_pi": q_pi.mean().item(), "mc_return": b["mc_return"].mean().item(),
                "calib_binds": (q_pi < mc).float().mean().item()}
        pushed = torch.maximum(q_pi, mc).view(-1, B, n).mean(-1)                       # (num_q, B)
        gap = pushed - q_data
        info["cql_gap"] = gap.mean().item()
        return gap.mean(-1).sum(), info

    def _calql_term(self, b, z, q_data, xi):
        """rl.calql.form = cqlh: the Cal-QL code base's variant -- CQL(H) (log-sum-exp over uniform random actions and
        policy actions) with policy-action Q-values lower-bounded by MC returns. Not the form written in the PLD paper.

        rl.calql.random selects where the "random" actions come from:
          full     : uniform over the whole action box [-1, 1]^d (standard CQL / Cal-QL)
          residual : a_b + U[-xi, xi]^d, i.e. the residual policy's support. These sit right next to the data
                     action and are not calibrated, so they drag Q(s, a_data) and the TD target down
                     (first XR1 run, alpha=5: Q(data) -4.5, TD target -7 for 0/1 rewards).
        All densities are w.r.t. the executed action a_bar (policy log-prob converted with -d*log(xi))."""
        cq = self.rl["calql"]
        n = int(cq["num_samples"])
        B = z.shape[0]
        rep = lambda x: x.repeat_interleave(n, 0)  # noqa: E731
        zr, ab = rep(z), rep(b["a_b"])
        d = self.act_dim
        if cq.get("random", "full") == "full":
            a_rand = torch.rand(B * n, d, device=self.device) * 2 - 1
            log_dens_rand = -d * math.log(2.0)
        else:
            a_rand = combine(ab, (torch.rand(B * n, d, device=self.device) * 2 - 1) * xi * self.xi_scale, d)
            log_dens_rand = -d * math.log(2 * xi)
        q_rand = self.critic(None, None, None, self._qa(ab, a_rand), z=zr)
        to_abar = -d * math.log(max(xi, 1e-8))
        with torch.no_grad():
            a_pi, logp_pi = self.actor(rep(b["feat"]), rep(b["proprio"]), ab, xi)
            a_nx, logp_nx = self.actor(rep(b["next_feat"]), rep(b["next_proprio"]), rep(b["next_a_b"]), xi)
            logp_pi, logp_nx = logp_pi + to_abar, logp_nx + to_abar
        q_pi = self.critic(None, None, None, self._qa(ab, combine(ab, a_pi, d)), z=zr)
        q_nx = self.critic(None, None, None, self._qa(ab, combine(ab, a_nx, d)), z=zr)
        mc = rep(b["mc_return"])[None]
        info = {"q_pi": q_pi.mean().item(), "q_rand": q_rand.mean().item(), "mc_return": b["mc_return"].mean().item(),
                "calib_binds": (q_pi < mc).float().mean().item()}
        q_pi = torch.maximum(q_pi, mc)   # Cal-QL calibration
        q_nx = torch.maximum(q_nx, mc)
        cat = torch.cat([
            (q_rand - log_dens_rand).view(-1, B, n),
            (q_pi - logp_pi[None]).view(-1, B, n),
            (q_nx - logp_nx[None]).view(-1, B, n),
        ], dim=-1)
        lse = torch.logsumexp(cat, dim=-1)  # (num_q, B)
        cql = (lse - q_data).mean(-1).sum()
        info["cql_gap"] = (lse - q_data).mean().item()
        return cql, info

    def actor_loss(self, b, xi):
        a_d, logp = self.actor(b["feat"], b["proprio"], b["a_b"], xi)
        with torch.no_grad():
            z = self.critic.encoder(b["feat"], b["proprio"], b["a_b"])
        q = self.critic(None, None, None, self._qa(b["a_b"], combine(b["a_b"], a_d, self.act_dim)), z=z).min(0).values
        alpha = self.log_alpha.exp().detach()
        loss = (alpha * logp - q).mean()
        alpha_loss = -(self.log_alpha * (logp.detach() + self.target_entropy)).mean()
        return loss, alpha_loss, {"entropy": -logp.mean().item(), "alpha": alpha.item()}

    # ------------------------------------------------------------------ updates
    def update(self, b_np: dict, calql_alpha: float = 0.0) -> dict:
        b = self._to_t(b_np)
        xi = self.xi
        closs, info = self.critic_loss(b, xi, calql_alpha)
        self._step(self.critic_opt, closs, self.critic.parameters())
        self._soft_update()
        self.n_updates += 1
        info["critic_loss"] = closs.item()
        if self.n_updates % int(self.rl["critic_actor_ratio"]) == 0:
            # critic params receive no grads here (q computed with frozen z, and actor_opt only steps the actor)
            aloss, alpha_loss, ainfo = self.actor_loss(b, xi)
            self._step(self.actor_opt, aloss, self.actor.parameters())
            self.alpha_opt.zero_grad(set_to_none=True)
            alpha_loss.backward()
            self.alpha_opt.step()
            info.update(ainfo, actor_loss=aloss.item())
        return info

    def reserve_gpu_memory(self, force: bool = False) -> None:
        """Run the largest forward/backward once on dummy data (no optimizer step, weights unchanged) so the CUDA
        allocator claims the learner's memory before the env renderers fill the GPU."""
        if self.device.type != "cuda" and not force:
            return
        a, B, rng = self._arch, self.batch, np.random.default_rng(0)
        feat = (a["n_cams"], *a["fmap_shape"])
        b = self._to_t({
            "feat": rng.standard_normal((B, *feat)).astype(np.float16), "next_feat": rng.standard_normal((B, *feat)).astype(np.float16),
            "proprio": np.zeros((B, a["proprio_dim"]), np.float32), "next_proprio": np.zeros((B, a["proprio_dim"]), np.float32),
            "a_b": np.zeros((B, a["base_dim"]), np.float32), "next_a_b": np.zeros((B, a["base_dim"]), np.float32),
            "a_bar": np.zeros((B, self.act_dim), np.float32), "reward": np.zeros(B, np.float32),
            "done": np.zeros(B, np.float32), "mc_return": np.zeros(B, np.float32),
        })
        closs, _ = self.critic_loss(b, self.xi, calql_alpha=1.0)
        closs.backward()
        aloss, _, _ = self.actor_loss(b, self.xi)
        aloss.backward()
        for opt in (self.critic_opt, self.actor_opt, self.alpha_opt):
            opt.zero_grad(set_to_none=True)
        # batches the prefetch threads keep on the device (queue + one in flight per thread)
        nt = int(self.rl.get("prefetch_threads", 0))
        if nt > 0:
            hold = [torch.empty((2, B, *feat), dtype=torch.float16, device=self.device)
                    for _ in range(nt + int(self.rl.get("prefetch_depth", 2)))]
            del hold

    def pretrain_calql(self, offline: ReplayBuffer, steps: int, log_fn=None):
        alpha = float(self.rl["calql"]["alpha"])
        nt = int(self.rl.get("prefetch_threads", 0))
        pf = BatchPrefetcher(offline, None, self.batch, self.device, threads=nt, depth=int(self.rl.get("prefetch_depth", 2)),
                             seed=int(self.rng.integers(2**31))) if nt > 0 else None
        try:
            for i in range(steps):
                b = pf.get() if pf else offline.sample(self.batch, self.rng)
                info = self.update(b, calql_alpha=alpha)
                if log_fn and (i % 1000 == 0 or i == steps - 1):
                    log_fn(step=i, **info)
        finally:
            if pf:
                pf.close()
        self.init_actor()  # Alg. 1: delta policy randomly initialised after critic warm-start
        self.n_updates = 0

    # ------------------------------------------------------------------ acting
    @torch.no_grad()
    def act(self, actor: ResidualActor, feats, proprio, a_base, deterministic: bool) -> np.ndarray:
        f = torch.as_tensor(feats, device=self.device)
        p = torch.as_tensor(proprio, device=self.device)
        ab = torch.as_tensor(a_base, device=self.device)
        a_d, _ = actor(f, p, ab, self.xi, deterministic=deterministic, with_logprob=False)
        return a_d.cpu().numpy()

    def state_dict(self) -> dict:
        return {
            "actor": self.actor.state_dict(), "critic": self.critic.state_dict(),
            "critic_targ": self.critic_targ.state_dict(), "log_alpha": self.log_alpha.detach().cpu(),
            "online_steps": self.online_steps, "xi_steps": self.xi_steps, "xi": self.xi,
            "n_updates": self.n_updates, "arch": self._arch,
        }

    def load_state_dict(self, sd: dict):
        self.actor.load_state_dict(sd["actor"])
        self.critic.load_state_dict(sd["critic"])
        self.critic_targ.load_state_dict(sd["critic_targ"])
        with torch.no_grad():
            self.log_alpha.copy_(sd["log_alpha"].to(self.device))
        self.online_steps = sd.get("online_steps", 0)
        self.xi_steps = sd.get("xi_steps", 0)  # stage 2 / eval continue with the xi the actor ended training at
        self.n_updates = sd.get("n_updates", 0)


class AsyncLearner(threading.Thread):
    """SERL-style asynchronous learner for one specialist. Throttled to `utd` critic updates per stored
    online transition; pushes actor weights to the collector copy every `actor_sync_every` updates."""

    def __init__(self, agent: ResidualSAC, offline: ReplayBuffer, online: ReplayBuffer, collector_actor,
                 actor_lock: threading.Lock, log_fn=None):
        super().__init__(daemon=True)
        self.agent, self.offline, self.online = agent, offline, online
        self.collector_actor, self.actor_lock = collector_actor, actor_lock
        self.log_fn = log_fn
        self.stop_flag = threading.Event()
        self.utd = float(agent.rl.get("utd", 1.0))
        self.sync_every = int(agent.rl.get("actor_sync_every", 50))
        self.error: BaseException | None = None

    def push_actor(self):
        with self.actor_lock:
            self.collector_actor.load_state_dict(self.agent.actor.state_dict())

    def run(self):
        nt = int(self.agent.rl.get("prefetch_threads", 0))
        pf = None
        try:
            last_log = time.time()
            while not self.stop_flag.is_set():
                budget = self.utd * self.online.total_added
                if len(self.online) == 0 or self.agent.n_updates >= budget:
                    time.sleep(0.005)
                    continue
                if nt > 0 and pf is None:   # started once the online buffer has data: every batch is half/half
                    pf = BatchPrefetcher(self.offline, self.online, self.agent.batch, self.agent.device, threads=nt,
                                         depth=int(self.agent.rl.get("prefetch_depth", 2)),
                                         seed=int(self.agent.rng.integers(2**31)))
                b = pf.get() if pf else symmetric_sample(self.offline, self.online, self.agent.batch, self.agent.rng)
                info = self.agent.update(b)
                if self.agent.n_updates % self.sync_every == 0:
                    self.push_actor()
                if self.log_fn and time.time() - last_log > 30:
                    self.log_fn(n_updates=self.agent.n_updates, **info)
                    last_log = time.time()
        except BaseException as e:  # noqa: BLE001
            self.error = e
            raise
        finally:
            if pf is not None:
                pf.close()

    def stop(self):
        self.stop_flag.set()
        self.join(timeout=60)
