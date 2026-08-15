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


CURRENT_AUX_TARGETS = 44


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
    # Legacy checkpoints use masked-mean pooling for value. New checkpoints can
    # opt into a learned query so value can attend selectively to schema-added
    # tokens without perturbing the policy trunk.
    value_attention: bool = False
    # Optional tactical readout: two learned queries (self/opponent) attend to
    # the frozen state tokens and predict the four real-time top-out intervals.
    # This leaves the shared trunk, policy, and WDL value path untouched.
    topout_attention: bool = False
    # Inference/training ablation: preserve each base placement's prior mass
    # when FASTEST and WAIT_FOR_EVENT variants are both present. This changes
    # no parameters and is therefore safe to toggle on legacy checkpoints.
    factor_timing_policy: bool = False
    # Diagnostic/calibration bias applied only to WAIT_FOR_EVENT within a
    # factorized FASTEST/WAIT pair. Zero preserves the checkpoint's raw
    # conditional preference; negative values suppress untrained waiting.
    timing_wait_logit_bias: float = 0.0
    # Dedicated conditional timing head. When enabled, matched FASTEST/WAIT
    # actions share the FASTEST base-placement logit exactly; this head predicts
    # only log-odds(WAIT/FASTEST). With timing actions absent, policy logits are
    # therefore bit-identical to the base placement policy.
    timing_head: bool = False
    timing_head_init_bias: float = -2.0
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


def factor_timing_policy_logits(logits: torch.Tensor, actions: torch.Tensor,
                                action_mask: torch.Tensor,
                                wait_logit_bias: float = 0.0) -> torch.Tensor:
    """Preserve base-placement prior mass while splitting FASTEST vs WAIT.

    For an adjacent matched pair with raw logits b (FASTEST) and w (WAIT),
    subtract softplus(w-b) from both. Their log-sum-exp then equals b while the
    within-pair logit difference w-b is unchanged. Timing branching therefore
    cannot inflate a placement merely because it has two action variants.
    """
    if logits.shape[1] < 2:
        return logits
    legal = action_mask > 0.5
    delay = torch.round(actions[..., 21] * 5.0).long().clamp(0, 5)
    base_delta = torch.cat(
        [actions[:, 1:, :21] - actions[:, :-1, :21],
         actions[:, 1:, 22:] - actions[:, :-1, 22:]], dim=-1
    )
    same_base = base_delta.abs().amax(dim=-1) <= 1e-6
    pair = (
        legal[:, :-1] & legal[:, 1:]
        & (delay[:, :-1] == 0) & (delay[:, 1:] == 5)
        & same_base
    )
    safe_logits = logits.masked_fill(~legal, 0.0)
    wait_bias = float(wait_logit_bias)
    adjusted = safe_logits + (delay == 5).to(logits.dtype) * wait_bias
    delta = adjusted[:, 1:] - adjusted[:, :-1]
    pair_penalty = F.softplus(delta) * pair.to(logits.dtype)
    penalty = torch.zeros_like(safe_logits)
    penalty[:, :-1] = penalty[:, :-1] + pair_penalty
    penalty[:, 1:] = penalty[:, 1:] + pair_penalty
    # The bias is meaningful only on the WAIT member of a matched pair. Avoid
    # perturbing any hypothetical unpaired delay action.
    pair_wait_bias = torch.zeros_like(safe_logits)
    pair_wait_bias[:, 1:] = pair.to(logits.dtype) * wait_bias
    return logits + pair_wait_bias - penalty


def apply_timing_head_logits(logits: torch.Tensor, policy_hidden: torch.Tensor,
                             actions: torch.Tensor, action_mask: torch.Tensor,
                             timing_out: nn.Linear) -> torch.Tensor:
    """Factor timing with a dedicated conditional head and fixed placement mass."""
    if logits.shape[1] < 2:
        return logits
    legal = action_mask > 0.5
    delay = torch.round(actions[..., 21] * 5.0).long().clamp(0, 5)
    base_delta = torch.cat(
        [actions[:, 1:, :21] - actions[:, :-1, :21],
         actions[:, 1:, 22:] - actions[:, :-1, 22:]], dim=-1
    )
    same_base = base_delta.abs().amax(dim=-1) <= 1e-6
    pair = (
        legal[:, :-1] & legal[:, 1:]
        & (delay[:, :-1] == 0) & (delay[:, 1:] == 5)
        & same_base
    )
    if not pair.any():
        return logits

    # Read timing context from the FASTEST member only, so the conditional head
    # never depends on the previously unseen nonzero delay feature.
    delta = timing_out(policy_hidden[:, :-1]).squeeze(-1)
    penalty = F.softplus(delta)
    fast_value = logits[:, :-1] - penalty
    wait_value = logits[:, :-1] + delta - penalty

    adjust = torch.zeros_like(logits)
    pair_f = pair.to(logits.dtype)
    adjust[:, :-1] = adjust[:, :-1] + pair_f * (fast_value - logits[:, :-1])
    adjust[:, 1:] = adjust[:, 1:] + pair_f * (wait_value - logits[:, 1:])
    return logits + adjust


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
        if cfg.timing_head:
            self.timing_out = nn.Linear(cfg.width, 1)
            nn.init.zeros_(self.timing_out.weight)
            nn.init.constant_(self.timing_out.bias, float(cfg.timing_head_init_bias))
        else:
            self.timing_out = None

        # WDL value head (spec 10.2) and auxiliary regressions.  The optional
        # learned-query pooler is deliberately value-only: it can be migrated
        # and trained while leaving the trunk and policy bit-identical.
        if cfg.value_attention:
            self.value_query = nn.Parameter(torch.randn(1, 1, cfg.width) * 0.02)
            self.value_attn = nn.MultiheadAttention(
                cfg.width, cfg.heads, dropout=cfg.dropout, batch_first=True
            )
            self.value_norm = RMSNorm(cfg.width)
        else:
            self.register_parameter("value_query", None)
            self.value_attn = None
            self.value_norm = None

        self.value_head = nn.Sequential(
            nn.Linear(cfg.width, cfg.width // 2), nn.SiLU(), nn.Linear(cfg.width // 2, 3)
        )
        self.aux_head = nn.Sequential(
            nn.Linear(cfg.width, cfg.width // 2),
            nn.SiLU(),
            nn.Linear(cfg.width // 2, cfg.aux_targets),
        )
        if cfg.topout_attention:
            self.topout_query = nn.Parameter(torch.randn(1, 2, cfg.width) * 0.02)
            self.topout_attn = nn.MultiheadAttention(
                cfg.width, cfg.heads, dropout=cfg.dropout, batch_first=True
            )
            self.topout_norm = RMSNorm(cfg.width)
            self.topout_out = nn.Linear(cfg.width, 4)
        else:
            self.register_parameter("topout_query", None)
            self.topout_attn = None
            self.topout_norm = None
            self.topout_out = None

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
        value_pooled = pooled
        if self.cfg.value_attention:
            assert self.value_query is not None
            assert self.value_attn is not None
            assert self.value_norm is not None
            value_q = self.value_query.expand(x.shape[0], -1, -1)
            value_attn, _ = self.value_attn(
                value_q, x, x, key_padding_mask=pad, need_weights=False
            )
            value_pooled = self.value_norm(value_q + value_attn).squeeze(1)

        q = self.action_in(actions)
        attn, _ = self.policy_attn(q, x, x, key_padding_mask=pad, need_weights=False)
        policy_hidden = self.policy_norm(q + attn)
        logits = self.policy_out(policy_hidden).squeeze(-1)
        # Illegal/padded actions can never be selected.
        logits = logits.masked_fill(action_mask < 0.5, float("-inf"))
        if self.cfg.timing_head:
            assert self.timing_out is not None
            logits = apply_timing_head_logits(
                logits, policy_hidden, actions, action_mask, self.timing_out
            )
        elif self.cfg.factor_timing_policy:
            logits = factor_timing_policy_logits(
                logits, actions, action_mask, self.cfg.timing_wait_logit_bias
            )

        aux_out = self.aux_head(pooled)
        if self.cfg.topout_attention:
            assert self.topout_query is not None
            assert self.topout_attn is not None
            assert self.topout_norm is not None
            assert self.topout_out is not None
            topout_q = self.topout_query.expand(x.shape[0], -1, -1)
            topout_attn, _ = self.topout_attn(
                topout_q, x, x, key_padding_mask=pad, need_weights=False
            )
            topout_logits = self.topout_out(
                self.topout_norm(topout_q + topout_attn)
            )  # [B, self/opponent, 4 real-time intervals]
            aux_out = aux_out.clone()
            self_indices = (6, 10, 14, 18)
            opponent_indices = (7, 11, 15, 19)
            for horizon, index in enumerate(self_indices):
                aux_out[:, index] = topout_logits[:, 0, horizon]
            for horizon, index in enumerate(opponent_indices):
                aux_out[:, index] = topout_logits[:, 1, horizon]

        return logits, self.value_head(value_pooled), aux_out

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


def tetraformer_s() -> TetraFormer:
    """The spec 9.5 TetraFormer-S: ~10-20M parameters."""
    return TetraFormer(TetraFormerConfig(width=256, layers=8, heads=8, ffn=768))


def tetraformer_dev() -> TetraFormer:
    """A small variant that trains at a usable rate on a CPU."""
    return TetraFormer(TetraFormerConfig(width=64, layers=2, heads=4, ffn=192))


_VS_REAL_ATTACK_INDICES = (4, 8, 12, 16)
_VS_REAL_GARBAGE_CLEARED_INDICES = (36, 37, 38, 39)
_VS_HORIZON_SECONDS = (1.0, 2.0, 4.0, 8.0)
_CANCELLATION_AUX_INDICES = tuple(range(44, 52))


def _vs_auxiliary_loss(aux, aux_target, aux_valid_mask):
    """MSE on cumulative short-horizon VS/100 derived from schema-v3 aux targets."""
    required_index = max(_VS_REAL_GARBAGE_CLEARED_INDICES)
    if aux.shape[1] <= required_index:
        raise ValueError("VS auxiliary loss requires schema-v3 targets")
    attack_index = torch.as_tensor(_VS_REAL_ATTACK_INDICES, device=aux.device, dtype=torch.long)
    cleared_index = torch.as_tensor(
        _VS_REAL_GARBAGE_CLEARED_INDICES, device=aux.device, dtype=torch.long
    )
    durations = aux.new_tensor(_VS_HORIZON_SECONDS)[None, :]
    prediction_pressure = aux.index_select(1, attack_index) + aux.index_select(1, cleared_index)
    target_pressure = aux_target.index_select(1, attack_index) + aux_target.index_select(1, cleared_index)
    interval_valid = (
        aux_valid_mask.index_select(1, attack_index)
        * aux_valid_mask.index_select(1, cleared_index)
    ).clamp(0.0, 1.0)
    prediction_rate = prediction_pressure.cumsum(dim=1) / durations
    target_rate = target_pressure.cumsum(dim=1) / durations
    cumulative_valid = torch.cumprod(interval_valid, dim=1)
    valid_count = cumulative_valid.sum()
    loss = (
        ((prediction_rate - target_rate).pow(2) * cumulative_valid).sum()
        / valid_count.clamp(min=1.0)
    )
    return loss, valid_count


def _cancellation_auxiliary_loss(aux, aux_target, aux_valid_mask):
    """Extra MSE emphasis on schema-v4 garbage-cancellation targets."""
    required_index = max(_CANCELLATION_AUX_INDICES)
    if aux.shape[1] <= required_index:
        raise ValueError("cancellation auxiliary loss requires schema-v4 targets")
    index = torch.as_tensor(
        _CANCELLATION_AUX_INDICES, device=aux.device, dtype=torch.long
    )
    prediction = aux.index_select(1, index)
    target = aux_target.index_select(1, index)
    valid = aux_valid_mask.index_select(1, index).clamp(0.0, 1.0)
    valid_count = valid.sum()
    loss = ((prediction - target).pow(2) * valid).sum() / valid_count.clamp(min=1.0)
    return loss, valid_count


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
    # Distillation temperature acts on the *teacher* visit distribution rather
    # than the model logits.  T < 1 sharpens the target while preserving its
    # action ordering; this is useful when soft-policy CE improves ranking but
    # makes the learned prior too flat for finite-budget MCTS.  Ranking/timing
    # losses below deliberately continue to see the original search target.
    policy_target = batch["policy_target"]
    policy_target_temperature = float(weights.get("policy_target_temperature", 1.0))
    if policy_target_temperature != 1.0:
        exponent = 1.0 / policy_target_temperature
        policy_target = policy_target.clamp_min(0.0).pow(exponent)
        policy_target = policy_target / policy_target.sum(-1, keepdim=True).clamp_min(1e-12)
    # Rows are already masked to -inf on padding, so the softmax ignores it;
    # multiplying by the target (zero on padding) keeps those terms out.
    policy_loss = -(policy_target * logp.nan_to_num(neginf=0.0)).sum(-1).mean()

    # Optional supervision on the action that search actually executed.  This
    # is distinct from the averaged visit distribution when multiple root
    # determinizations vote for their Gumbel survivors. Pre-v4 datasets expose
    # chosen_action=-1 and therefore contribute nothing to this term.
    chosen_action_loss = policy_loss * 0.0
    chosen_disagreement_loss = policy_loss * 0.0
    chosen_action_weight = float(weights.get("chosen_action", 0.0))
    chosen_disagreement_weight = float(weights.get("chosen_disagreement", 0.0))
    chosen_action = batch.get("chosen_action")
    if (chosen_action_weight != 0.0 or chosen_disagreement_weight != 0.0) and chosen_action is not None:
        chosen = chosen_action.long()
        in_range = (chosen >= 0) & (chosen < logits.shape[1])
        safe = chosen.clamp(0, max(0, logits.shape[1] - 1))
        legal = batch["action_mask"].gather(1, safe[:, None]).squeeze(1) > 0.5
        valid_chosen = in_range & legal
        if chosen_action_weight != 0.0 and valid_chosen.any():
            chosen_action_loss = F.cross_entropy(logits[valid_chosen], safe[valid_chosen])
        if chosen_disagreement_weight != 0.0:
            # Policy-improvement distillation: only train positions where search
            # actually overturns the current policy argmax.  Replaying the ~80%
            # of already-agreeing states otherwise mostly changes calibration
            # while contributing no new action decision.
            model_best = logits.detach().argmax(dim=-1)
            disagreement = valid_chosen & (model_best != safe)
            if disagreement.any():
                chosen_disagreement_loss = F.cross_entropy(
                    logits[disagreement], safe[disagreement]
                )

    # Optional ranking supervision for the overall policy.  Soft visit-distribution
    # CE can improve while slightly degrading the ordering of the best few moves,
    # which is especially costly for low-budget Gumbel search.  Only confident
    # teacher preferences contribute strongly: near-ties between the best two
    # actions receive almost no rank weight.
    policy_rank_loss = policy_loss * 0.0
    policy_pair_rank_loss = policy_loss * 0.0
    policy_rank_weight = float(weights.get("policy_rank", 0.0))
    policy_pair_rank_weight = float(weights.get("policy_pair_rank", 0.0))
    if (policy_rank_weight != 0.0 or policy_pair_rank_weight != 0.0) and logits.shape[1] >= 2:
        target = batch["policy_target"]
        top2 = target.topk(2, dim=-1)
        teacher_best = top2.indices[:, 0]
        teacher_second = top2.indices[:, 1]
        confidence = (top2.values[:, 0] - top2.values[:, 1]).clamp_min(0.0)
        confidence_sum = confidence.sum()
        if confidence_sum > 0.0:
            if policy_rank_weight != 0.0:
                hard_ce = F.cross_entropy(logits, teacher_best, reduction="none")
                policy_rank_loss = (
                    (hard_ce * confidence).sum() / confidence_sum.clamp_min(1e-12)
                )
            if policy_pair_rank_weight != 0.0:
                best_logit = logits.gather(1, teacher_best[:, None]).squeeze(1)
                second_logit = logits.gather(1, teacher_second[:, None]).squeeze(1)
                pair_bce = F.softplus(-(best_logit - second_logit))
                policy_pair_rank_loss = (
                    (pair_bce * confidence).sum() / confidence_sum.clamp_min(1e-12)
                )

    # Optional experimental timing supervision. A WAIT_FOR_EVENT variant, when
    # present, is emitted immediately after its FASTEST base placement. Some
    # placements now need no WAIT because their base execution already reaches
    # the event, so do not assume the whole action list has even/odd pairs.
    # Full policy CE can fit aggregate timing mass while still learning the
    # wrong conditional choice inside individual placement pairs, so compare
    # matched adjacent logits directly against the teacher's within-pair ratio.
    timing_pair_loss = policy_loss * 0.0
    timing_rank_loss = policy_loss * 0.0
    timing_pair_weight = float(weights.get("timing_pair", 0.0))
    timing_rank_weight = float(weights.get("timing_rank", 0.0))
    if (timing_pair_weight != 0.0 or timing_rank_weight != 0.0) and logits.shape[1] >= 2:
        actions = batch["actions"]
        action_mask = batch["action_mask"] > 0.5
        delay = torch.round(actions[..., 21] * 5.0).long().clamp(0, 5)
        # Features other than delay_bin must identify the same base placement.
        base_delta = torch.cat(
            [actions[:, 1:, :21] - actions[:, :-1, :21],
             actions[:, 1:, 22:] - actions[:, :-1, 22:]], dim=-1
        )
        same_base = base_delta.abs().amax(dim=-1) <= 1e-6
        fast_logits = logits[:, :-1]
        wait_logits = logits[:, 1:]
        fast_target = batch["policy_target"][:, :-1]
        wait_target = batch["policy_target"][:, 1:]
        pair_mass = fast_target + wait_target
        valid_pair = (
            action_mask[:, :-1]
            & action_mask[:, 1:]
            & (delay[:, :-1] == 0)
            & (delay[:, 1:] == 5)
            & same_base
            & (pair_mass > 0.0)
        )
        if valid_pair.any():
            teacher_wait = wait_target / pair_mass.clamp_min(1e-12)
            pair_delta = (wait_logits - fast_logits)[valid_pair]
            pair_teacher = teacher_wait[valid_pair]
            pair_weights = pair_mass[valid_pair]
            pair_bce = F.binary_cross_entropy_with_logits(
                pair_delta, pair_teacher, reduction="none"
            )
            timing_pair_loss = (pair_bce * pair_weights).sum() / pair_weights.sum().clamp_min(1e-12)

            # Ranking variant: train only which side of the pair the teacher
            # prefers, while downweighting near-ties by visit-margin confidence.
            pair_label = (pair_teacher > 0.5).float()
            pair_confidence = (2.0 * pair_teacher - 1.0).abs()
            rank_weights = pair_weights * pair_confidence
            if rank_weights.sum() > 0.0:
                rank_bce = F.binary_cross_entropy_with_logits(
                    pair_delta, pair_label, reduction="none"
                )
                timing_rank_loss = (
                    (rank_bce * rank_weights).sum() /
                    rank_weights.sum().clamp_min(1e-12)
                )

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

    # Optional VS-style pressure supervision. Schema v3 stores disjoint
    # real-time attack and garbage-cleared counts; cumulative sums over those
    # intervals recover (sent + garbage-cleared) / seconds = VS / 100 at
    # 1/2/4/8-second horizons without widening the dataset schema.
    vs_aux_loss = aux_loss * 0.0
    vs_aux_valid = aux_loss.new_zeros(())
    vs_aux_weight = float(weights.get("vs_aux", 0.0))
    if aux.shape[1] > max(_VS_REAL_GARBAGE_CLEARED_INDICES):
        vs_aux_loss, vs_aux_valid = _vs_auxiliary_loss(aux, aux_target, aux_valid_mask)
    elif vs_aux_weight != 0.0:
        raise ValueError("VS auxiliary loss requires schema-v3 targets")

    # Optional extra emphasis on the eight schema-v4 garbage-cancellation
    # channels. They are already included in the generic auxiliary MSE; this
    # dedicated term lets experiments strengthen cancellation representation
    # without changing the primary game reward or the other auxiliary targets.
    cancellation_aux_loss = aux_loss * 0.0
    cancellation_aux_valid = aux_loss.new_zeros(())
    cancellation_aux_weight = float(weights.get("cancellation_aux", 0.0))
    if aux.shape[1] > max(_CANCELLATION_AUX_INDICES):
        cancellation_aux_loss, cancellation_aux_valid = _cancellation_auxiliary_loss(
            aux, aux_target, aux_valid_mask
        )
    elif cancellation_aux_weight != 0.0:
        raise ValueError("cancellation auxiliary loss requires schema-v4 targets")

    # Dedicated binary supervision for the four disjoint real-time top-out
    # intervals (self/opponent). These targets are sparse and were previously
    # diluted among dozens of MSE regressions. The indices remain stable across
    # aux schema v2/v3 because v3 only appends garbage-clear channels.
    topout_aux_loss = aux_loss * 0.0
    topout_aux_weight = float(weights.get("topout_aux", 0.0))
    topout_indices = (6, 7, 10, 11, 14, 15, 18, 19)
    if topout_aux_weight != 0.0 and aux.shape[1] > max(topout_indices):
        index = torch.as_tensor(topout_indices, device=aux.device, dtype=torch.long)
        topout_logits = aux.index_select(1, index)
        topout_target = aux_target.index_select(1, index).clamp(0.0, 1.0)
        topout_valid = aux_valid_mask.index_select(1, index)
        topout_error = F.binary_cross_entropy_with_logits(
            topout_logits, topout_target, reduction="none"
        )
        topout_aux_loss = (
            (topout_error * topout_valid).sum() /
            topout_valid.sum().clamp(min=1.0)
        )

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
        + chosen_action_weight * chosen_action_loss
        + chosen_disagreement_weight * chosen_disagreement_loss
        + weights["value"] * value_loss
        + weights["aux"] * aux_loss
        + vs_aux_weight * vs_aux_loss
        + cancellation_aux_weight * cancellation_aux_loss
        + topout_aux_weight * topout_aux_loss
        + policy_rank_weight * policy_rank_loss
        + policy_pair_rank_weight * policy_pair_rank_loss
        + timing_pair_weight * timing_pair_loss
        + timing_rank_weight * timing_rank_loss
    )
    return total, {
        "policy": policy_loss.item(),
        "chosen_action": chosen_action_loss.item(),
        "chosen_disagreement": chosen_disagreement_loss.item(),
        "policy_rank": policy_rank_loss.item(),
        "policy_pair_rank": policy_pair_rank_loss.item(),
        "value": value_loss.item(),
        "value_accuracy": value_accuracy.item(),
        "value_scalar_mse": value_scalar_mse.item(),
        "aux": aux_loss.item(),
        "vs_aux": vs_aux_loss.item(),
        "vs_aux_valid": vs_aux_valid.item(),
        "cancellation_aux": cancellation_aux_loss.item(),
        "cancellation_aux_valid": cancellation_aux_valid.item(),
        "topout_aux": topout_aux_loss.item(),
        "timing_pair": timing_pair_loss.item(),
        "timing_rank": timing_rank_loss.item(),
        "aux_valid": valid_count.item(),
        "aux_prediction_mean": prediction_mean.item(),
        "aux_prediction_variance": prediction_variance.item(),
        "aux_per_target": aux_per_target.detach().cpu().tolist(),
        # Kept private-ish for the training loop's gradient diagnostics.  The
        # public scalar entries above remain plain Python numbers for callers.
        "_loss_tensors": (
            policy_loss, value_loss, aux_loss, vs_aux_loss, cancellation_aux_loss,
            timing_pair_loss, timing_rank_loss
        ),
        "total": total.item(),
    }
