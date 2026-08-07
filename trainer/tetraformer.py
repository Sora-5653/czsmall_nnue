# SPDX-License-Identifier: MIT
"""TetraFormer: the encoder-only policy/value network from spec sections 9-10.

Architecture, following the spec:

* state tokens are embedded and passed through a pre-norm Transformer encoder
  (spec 9.5: RMSNorm, SwiGLU, configurable width/depth),
* padding is masked everywhere, so a short position is not attended to,
* the policy head is **variable length** (spec 10.1): each legal action becomes
  a query that cross-attends to the state tokens, producing one logit per legal
  action rather than a fixed x/rotation grid,
* the value head is WDL (spec 10.2), plus auxiliary regression heads.

Sizes are configurable; `tetraformer_s()` matches the spec's TetraFormer-S and
`tetraformer_dev()` is a small variant that trains at a usable speed on CPU.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


CURRENT_AUX_TARGETS = 36


@dataclass
class TetraFormerConfig:
    token_features: int = 24
    action_features: int = 24
    width: int = 256
    layers: int = 8
    heads: int = 8
    ffn: int = 768
    # New models consume the version-2 interval target contract.  Legacy
    # checkpoints keep their explicit value (normally 4) when restored.
    aux_targets: int = CURRENT_AUX_TARGETS
    dropout: float = 0.0


class RMSNorm(nn.Module):
    """Pre-norm RMSNorm (spec 9.5)."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return x * rms * self.weight


class SwiGLU(nn.Module):
    """Gated feed-forward block (spec 9.5)."""

    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.gate = nn.Linear(dim, hidden, bias=False)
        self.up = nn.Linear(dim, hidden, bias=False)
        self.down = nn.Linear(hidden, dim, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class EncoderBlock(nn.Module):
    def __init__(self, cfg: TetraFormerConfig):
        super().__init__()
        self.n1 = RMSNorm(cfg.width)
        self.attn = nn.MultiheadAttention(
            cfg.width, cfg.heads, dropout=cfg.dropout, batch_first=True
        )
        self.n2 = RMSNorm(cfg.width)
        self.ffn = SwiGLU(cfg.width, cfg.ffn)

    def forward(self, x, key_padding_mask):
        h = self.n1(x)
        attn, _ = self.attn(h, h, h, key_padding_mask=key_padding_mask, need_weights=False)
        x = x + attn
        return x + self.ffn(self.n2(x))


class TetraFormer(nn.Module):
    def __init__(self, cfg: TetraFormerConfig = TetraFormerConfig()):
        super().__init__()
        self.cfg = cfg
        self.token_in = nn.Linear(cfg.token_features, cfg.width)
        self.blocks = nn.ModuleList([EncoderBlock(cfg) for _ in range(cfg.layers)])
        self.norm = RMSNorm(cfg.width)

        # Variable-length policy head (spec 10.1): action queries cross-attend
        # to the encoded state.
        self.action_in = nn.Linear(cfg.action_features, cfg.width)
        self.policy_attn = nn.MultiheadAttention(
            cfg.width, cfg.heads, dropout=cfg.dropout, batch_first=True
        )
        self.policy_norm = RMSNorm(cfg.width)
        self.policy_out = nn.Linear(cfg.width, 1)

        # WDL value head (spec 10.2) and auxiliary regressions.
        self.value_head = nn.Sequential(
            nn.Linear(cfg.width, cfg.width // 2), nn.SiLU(), nn.Linear(cfg.width // 2, 3)
        )
        self.aux_head = nn.Sequential(
            nn.Linear(cfg.width, cfg.width // 2),
            nn.SiLU(),
            nn.Linear(cfg.width // 2, cfg.aux_targets),
        )

    def forward(self, tokens, token_mask, actions, action_mask):
        """Returns (policy_logits [B, A], wdl_logits [B, 3], aux [B, aux])."""
        # MultiheadAttention wants True where a position should be IGNORED.
        pad = token_mask < 0.5

        x = self.token_in(tokens)
        for blk in self.blocks:
            x = blk(x, pad)
        x = self.norm(x)

        # Masked mean pooling for the value head: padding must not dilute it.
        w = token_mask.unsqueeze(-1)
        pooled = (x * w).sum(1) / w.sum(1).clamp(min=1.0)

        q = self.action_in(actions)
        attn, _ = self.policy_attn(q, x, x, key_padding_mask=pad, need_weights=False)
        logits = self.policy_out(self.policy_norm(q + attn)).squeeze(-1)
        # Illegal/padded actions can never be selected.
        logits = logits.masked_fill(action_mask < 0.5, float("-inf"))

        return logits, self.value_head(pooled), self.aux_head(pooled)

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


def tetraformer_s() -> TetraFormer:
    """The spec 9.5 TetraFormer-S: ~10-20M parameters."""
    return TetraFormer(TetraFormerConfig(width=256, layers=8, heads=8, ffn=768))


def tetraformer_dev() -> TetraFormer:
    """A small variant that trains at a usable rate on a CPU."""
    return TetraFormer(TetraFormerConfig(width=64, layers=2, heads=4, ffn=192))


def losses(model, batch, weights=None):
    """Total loss and its components (spec 13.5).

    Policy is cross-entropy against the search visit distribution, value is
    cross-entropy against the WDL one-hot implied by the game result, and the
    auxiliary heads are plain MSE.
    """
    weights = weights or {"policy": 1.0, "value": 1.0, "aux": 0.1}

    logits, wdl, aux = model(
        batch["tokens"], batch["token_mask"], batch["actions"], batch["action_mask"]
    )

    logp = torch.log_softmax(logits, dim=-1)
    # Rows are already masked to -inf on padding, so the softmax ignores it;
    # multiplying by the target (zero on padding) keeps those terms out.
    policy_loss = -(batch["policy_target"] * logp.nan_to_num(neginf=0.0)).sum(-1).mean()

    # Map z in [-1, 1] onto WDL targets.
    z = batch["value_target"]
    win = (z > 0.5).float()
    loss_ = (z < -0.5).float()
    draw = 1.0 - win - loss_
    value_target = torch.stack([win, draw, loss_], dim=-1)
    value_loss = -(value_target * torch.log_softmax(wdl, dim=-1)).sum(-1).mean()

    aux_target = batch["aux_target"]
    aux_valid_mask = batch.get("aux_valid_mask")
    if aux_valid_mask is None:
        aux_valid_mask = torch.ones_like(aux_target)
    aux_error = (aux - aux_target).pow(2)
    valid_count = aux_valid_mask.sum().clamp(min=1.0)
    aux_loss = (aux_error * aux_valid_mask).sum() / valid_count
    aux_per_target = (
        (aux_error * aux_valid_mask).sum(dim=0) /
        aux_valid_mask.sum(dim=0).clamp(min=1.0)
    )

    # Diagnostics are kept separate from the optimised value loss.  In
    # particular, value_accuracy makes it obvious when the WDL head is still
    # at chance even if the policy loss is decreasing.
    with torch.no_grad():
        value_prob = torch.softmax(wdl, dim=-1)
        value_class = value_target.argmax(dim=-1)
        value_accuracy = (value_prob.argmax(dim=-1) == value_class).float().mean()
        value_scalar = value_prob[:, 0] - value_prob[:, 2]
        value_scalar_mse = F.mse_loss(value_scalar, z)
        valid_aux = aux_valid_mask > 0.5
        prediction_mean = aux[valid_aux].mean() if valid_aux.any() else aux.new_zeros(())
        prediction_variance = aux[valid_aux].var(unbiased=False) if valid_aux.any() else aux.new_zeros(())

    total = (
        weights["policy"] * policy_loss
        + weights["value"] * value_loss
        + weights["aux"] * aux_loss
    )
    return total, {
        "policy": policy_loss.item(),
        "value": value_loss.item(),
        "value_accuracy": value_accuracy.item(),
        "value_scalar_mse": value_scalar_mse.item(),
        "aux": aux_loss.item(),
        "aux_valid": valid_count.item(),
        "aux_prediction_mean": prediction_mean.item(),
        "aux_prediction_variance": prediction_variance.item(),
        "aux_per_target": aux_per_target.detach().cpu().tolist(),
        # Kept private-ish for the training loop's gradient diagnostics.  The
        # public scalar entries above remain plain Python numbers for callers.
        "_loss_tensors": (policy_loss, value_loss, aux_loss),
        "total": total.item(),
    }
