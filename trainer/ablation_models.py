# SPDX-License-Identifier: MIT
"""CNN and CNN+Transformer ablation models for the fixed TetraFormer input contract.

These models intentionally keep the public forward signature identical to TetraFormer:
    (tokens, token_mask, actions, action_mask) -> (policy_logits, wdl_logits, aux)

The tokenizer stores exact 10-wide occupancy bits in row-token features 8..17.
For the current two-player contract, self row tokens are the first 24 tokens and the
opponent board occupies the last 36 real tokens (24 rows + 10 columns + summary +
opponent counters).  This lets the CNN recover both 24x10 occupancy grids without
changing the dataset or C++ engine.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from tetraformer import EncoderBlock, RMSNorm, TetraFormer, TetraFormerConfig


@dataclass
class CNNConfig:
    token_features: int = 24
    action_features: int = 24
    aux_targets: int = 36
    width: int = 256
    board_rows: int = 24
    board_cols: int = 10
    cnn_channels: int = 192
    cnn_blocks: int = 10


@dataclass
class HybridConfig:
    token_features: int = 24
    action_features: int = 24
    aux_targets: int = 36
    width: int = 256
    layers: int = 8
    heads: int = 8
    ffn: int = 768
    board_rows: int = 24
    board_cols: int = 10
    cnn_channels: int = 64
    cnn_blocks: int = 4
    dropout: float = 0.0


@dataclass
class SplitHybridConfig:
    token_features: int = 24
    action_features: int = 24
    aux_targets: int = 36
    width: int = 256
    layers: int = 8
    heads: int = 8
    ffn: int = 768
    board_rows: int = 24
    board_cols: int = 10
    cnn_channels: int = 64
    cnn_blocks: int = 4
    dropout: float = 0.0


def extract_board_planes(tokens: torch.Tensor, token_mask: torch.Tensor,
                         rows: int = 24, cols: int = 10) -> torch.Tensor:
    """Recover [self, opponent, difference, y, x] board planes from token features."""
    if tokens.ndim != 3:
        raise ValueError(f"tokens must be [B,T,F], got {tuple(tokens.shape)}")
    if tokens.shape[1] < rows or tokens.shape[2] < 8 + cols:
        raise ValueError("token tensor is too small for the board row contract")

    batch, token_count, features = tokens.shape
    self_board = tokens[:, :rows, 8:8 + cols]

    real_counts = token_mask.sum(dim=1).round().long()
    # Two-player encoding ends with 24 opponent rows, 10 columns, one summary,
    # and one opponent-counters token.  Single-board/missing-opponent positions
    # are conservatively mapped to an all-zero opponent plane.
    opponent_start = real_counts - 36
    offsets = torch.arange(rows, device=tokens.device).view(1, rows)
    gather_index = opponent_start.view(batch, 1) + offsets
    safe_index = gather_index.clamp(min=0, max=max(0, token_count - 1))
    gathered = tokens.gather(
        1, safe_index.unsqueeze(-1).expand(batch, rows, features)
    )
    opponent_board = gathered[:, :, 8:8 + cols]
    has_opponent = (opponent_start >= rows).view(batch, 1, 1)
    opponent_board = torch.where(has_opponent, opponent_board, torch.zeros_like(opponent_board))

    y = torch.linspace(0.0, 1.0, rows, device=tokens.device, dtype=tokens.dtype)
    y = y.view(1, rows, 1).expand(batch, rows, cols)
    x = torch.linspace(0.0, 1.0, cols, device=tokens.device, dtype=tokens.dtype)
    x = x.view(1, 1, cols).expand(batch, rows, cols)

    # Conv2d expects [B,C,H,W].  Difference is a cheap relational plane while
    # coordinate channels prevent global pooling from erasing absolute height.
    return torch.stack(
        [self_board, opponent_board, self_board - opponent_board, y, x], dim=1
    )


class PatchConv2d(nn.Module):
    """3x3 shared convolution expressed as unfold + GEMM.

    The Windows ROCm/MIOpen build currently used by this workspace can select a
    CK grouped-convolution backward kernel that is invalid on gfx1201.  This is
    mathematically the same spatially shared 3x3 convolution, but it stays on
    PyTorch unfold + matrix multiplication and therefore avoids that MIOpen path.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels * 9))
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, _, height, width = x.shape
        patches = F.unfold(x, kernel_size=3, padding=1).transpose(1, 2)
        out = patches.matmul(self.weight.t())
        return out.transpose(1, 2).reshape(batch, self.out_channels, height, width)


class ResidualConvBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(1, channels)
        self.conv1 = PatchConv2d(channels, channels)
        self.norm2 = nn.GroupNorm(1, channels)
        self.conv2 = PatchConv2d(channels, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = self.conv2(F.silu(self.norm2(h)))
        return x + h


class BoardCNN(nn.Module):
    def __init__(self, width: int, channels: int, blocks: int, rows: int, cols: int):
        super().__init__()
        self.rows = rows
        self.cols = cols
        self.stem = PatchConv2d(5, channels)
        self.blocks = nn.ModuleList([ResidualConvBlock(channels) for _ in range(blocks)])
        self.norm = nn.GroupNorm(1, channels)
        self.out = nn.Linear(channels * 2, width)

    def forward(self, tokens: torch.Tensor, token_mask: torch.Tensor) -> torch.Tensor:
        x = extract_board_planes(tokens, token_mask, self.rows, self.cols)
        x = self.stem(x)
        for block in self.blocks:
            x = block(x)
        x = F.silu(self.norm(x))
        avg = x.mean(dim=(-2, -1))
        peak = x.amax(dim=(-2, -1))
        return self.out(torch.cat([avg, peak], dim=-1))


class CNNPolicyValue(nn.Module):
    """Pure CNN board encoder + permutation-invariant global-token MLP baseline."""

    architecture = "cnn"

    def __init__(self, cfg: CNNConfig):
        super().__init__()
        self.cfg = cfg
        self.board = BoardCNN(
            cfg.width, cfg.cnn_channels, cfg.cnn_blocks, cfg.board_rows, cfg.board_cols
        )
        self.token_in = nn.Linear(cfg.token_features, cfg.width)
        self.token_norm = RMSNorm(cfg.width)
        self.state_fuse = nn.Sequential(
            nn.Linear(cfg.width * 2, cfg.width), nn.SiLU(), nn.Linear(cfg.width, cfg.width)
        )
        self.state_norm = RMSNorm(cfg.width)

        self.action_in = nn.Linear(cfg.action_features, cfg.width)
        self.action_state = nn.Linear(cfg.width, cfg.width, bias=False)
        self.policy_hidden = nn.Linear(cfg.width, cfg.width)
        self.policy_norm = RMSNorm(cfg.width)
        self.policy_out = nn.Linear(cfg.width, 1)

        self.value_head = nn.Sequential(
            nn.Linear(cfg.width, cfg.width // 2), nn.SiLU(), nn.Linear(cfg.width // 2, 3)
        )
        self.aux_head = nn.Sequential(
            nn.Linear(cfg.width, cfg.width // 2),
            nn.SiLU(),
            nn.Linear(cfg.width // 2, cfg.aux_targets),
        )

    def forward(self, tokens, token_mask, actions, action_mask):
        board = self.board(tokens, token_mask)
        token_h = self.token_norm(self.token_in(tokens))
        weight = token_mask.unsqueeze(-1)
        pooled = (token_h * weight).sum(dim=1) / weight.sum(dim=1).clamp(min=1.0)
        state = self.state_norm(self.state_fuse(torch.cat([board, pooled], dim=-1)))

        query = self.action_in(actions)
        conditioned = query + self.action_state(state).unsqueeze(1) + query * state.unsqueeze(1)
        hidden = self.policy_norm(self.policy_hidden(F.silu(conditioned)))
        logits = self.policy_out(hidden).squeeze(-1)
        logits = logits.masked_fill(action_mask < 0.5, float("-inf"))
        return logits, self.value_head(state), self.aux_head(state)

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


class CNNTransformerHybrid(nn.Module):
    """TetraFormer-S trunk with one learned CNN board token appended to state tokens."""

    architecture = "hybrid"

    def __init__(self, cfg: HybridConfig):
        super().__init__()
        self.cfg = cfg
        tf_cfg = TetraFormerConfig(
            token_features=cfg.token_features,
            action_features=cfg.action_features,
            width=cfg.width,
            layers=cfg.layers,
            heads=cfg.heads,
            ffn=cfg.ffn,
            aux_targets=cfg.aux_targets,
            dropout=cfg.dropout,
        )
        self.board = BoardCNN(
            cfg.width, cfg.cnn_channels, cfg.cnn_blocks, cfg.board_rows, cfg.board_cols
        )
        self.token_in = nn.Linear(cfg.token_features, cfg.width)
        self.blocks = nn.ModuleList([EncoderBlock(tf_cfg) for _ in range(cfg.layers)])
        self.norm = RMSNorm(cfg.width)

        self.action_in = nn.Linear(cfg.action_features, cfg.width)
        self.policy_attn = nn.MultiheadAttention(
            cfg.width, cfg.heads, dropout=cfg.dropout, batch_first=True
        )
        self.policy_norm = RMSNorm(cfg.width)
        self.policy_out = nn.Linear(cfg.width, 1)

        self.value_head = nn.Sequential(
            nn.Linear(cfg.width, cfg.width // 2), nn.SiLU(), nn.Linear(cfg.width // 2, 3)
        )
        self.aux_head = nn.Sequential(
            nn.Linear(cfg.width, cfg.width // 2),
            nn.SiLU(),
            nn.Linear(cfg.width // 2, cfg.aux_targets),
        )

    def forward(self, tokens, token_mask, actions, action_mask):
        board_token = self.board(tokens, token_mask).unsqueeze(1)
        x = torch.cat([self.token_in(tokens), board_token], dim=1)
        board_mask = torch.ones(
            (token_mask.shape[0], 1), device=token_mask.device, dtype=token_mask.dtype
        )
        extended_mask = torch.cat([token_mask, board_mask], dim=1)
        pad = extended_mask < 0.5
        for block in self.blocks:
            x = block(x, pad)
        x = self.norm(x)

        weight = extended_mask.unsqueeze(-1)
        pooled = (x * weight).sum(dim=1) / weight.sum(dim=1).clamp(min=1.0)
        query = self.action_in(actions)
        attended, _ = self.policy_attn(query, x, x, key_padding_mask=pad, need_weights=False)
        logits = self.policy_out(self.policy_norm(query + attended)).squeeze(-1)
        logits = logits.masked_fill(action_mask < 0.5, float("-inf"))
        return logits, self.value_head(pooled), self.aux_head(pooled)

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


class FusionCNNTransformer(nn.Module):
    """Shared CNN board embedding used by both Transformer policy and value.

    Compared with `CNNTransformerHybrid`, the board embedding is still appended
    as a Transformer token for policy, but value/aux also receive it directly
    instead of relying on masked mean pooling to preserve local information.
    This keeps CNN features jointly regularised by policy, value, and aux losses.
    """

    architecture = "fusion_hybrid"

    def __init__(self, cfg: HybridConfig):
        super().__init__()
        self.cfg = cfg
        tf_cfg = TetraFormerConfig(
            token_features=cfg.token_features,
            action_features=cfg.action_features,
            width=cfg.width,
            layers=cfg.layers,
            heads=cfg.heads,
            ffn=cfg.ffn,
            aux_targets=cfg.aux_targets,
            dropout=cfg.dropout,
        )
        self.board = BoardCNN(
            cfg.width, cfg.cnn_channels, cfg.cnn_blocks, cfg.board_rows, cfg.board_cols
        )
        self.token_in = nn.Linear(cfg.token_features, cfg.width)
        self.blocks = nn.ModuleList([EncoderBlock(tf_cfg) for _ in range(cfg.layers)])
        self.norm = RMSNorm(cfg.width)

        self.action_in = nn.Linear(cfg.action_features, cfg.width)
        self.policy_attn = nn.MultiheadAttention(
            cfg.width, cfg.heads, dropout=cfg.dropout, batch_first=True
        )
        self.policy_norm = RMSNorm(cfg.width)
        self.policy_out = nn.Linear(cfg.width, 1)

        self.value_fuse = nn.Sequential(
            nn.Linear(cfg.width * 2, cfg.width), nn.SiLU(), nn.Linear(cfg.width, cfg.width)
        )
        self.value_state_norm = RMSNorm(cfg.width)
        self.value_head = nn.Sequential(
            nn.Linear(cfg.width, cfg.width // 2), nn.SiLU(), nn.Linear(cfg.width // 2, 3)
        )
        self.aux_head = nn.Sequential(
            nn.Linear(cfg.width, cfg.width // 2),
            nn.SiLU(),
            nn.Linear(cfg.width // 2, cfg.aux_targets),
        )

    def forward(self, tokens, token_mask, actions, action_mask):
        board = self.board(tokens, token_mask)
        x = torch.cat([self.token_in(tokens), board.unsqueeze(1)], dim=1)
        board_mask = torch.ones(
            (token_mask.shape[0], 1), device=token_mask.device, dtype=token_mask.dtype
        )
        extended_mask = torch.cat([token_mask, board_mask], dim=1)
        pad = extended_mask < 0.5
        for block in self.blocks:
            x = block(x, pad)
        x = self.norm(x)

        weight = extended_mask.unsqueeze(-1)
        pooled = (x * weight).sum(dim=1) / weight.sum(dim=1).clamp(min=1.0)
        query = self.action_in(actions)
        attended, _ = self.policy_attn(query, x, x, key_padding_mask=pad, need_weights=False)
        logits = self.policy_out(self.policy_norm(query + attended)).squeeze(-1)
        logits = logits.masked_fill(action_mask < 0.5, float("-inf"))

        value_state = self.value_state_norm(
            self.value_fuse(torch.cat([pooled, board], dim=-1))
        )
        return logits, self.value_head(value_state), self.aux_head(value_state)

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


class DualPolicyCNNTransformer(nn.Module):
    """Transformer and CNN policy logits are averaged; CNN state also serves value.

    This forces the local CNN representation to receive direct policy gradients,
    matching the regularisation mechanism that appears important in the pure CNN
    baseline, while retaining a full Transformer policy path.
    """

    architecture = "dual_policy_hybrid"

    def __init__(self, cfg: HybridConfig):
        super().__init__()
        self.cfg = cfg
        tf_cfg = TetraFormerConfig(
            token_features=cfg.token_features,
            action_features=cfg.action_features,
            width=cfg.width,
            layers=cfg.layers,
            heads=cfg.heads,
            ffn=cfg.ffn,
            aux_targets=cfg.aux_targets,
            dropout=cfg.dropout,
        )

        self.token_in = nn.Linear(cfg.token_features, cfg.width)
        self.blocks = nn.ModuleList([EncoderBlock(tf_cfg) for _ in range(cfg.layers)])
        self.norm = RMSNorm(cfg.width)
        self.action_in = nn.Linear(cfg.action_features, cfg.width)
        self.policy_attn = nn.MultiheadAttention(
            cfg.width, cfg.heads, dropout=cfg.dropout, batch_first=True
        )
        self.policy_norm = RMSNorm(cfg.width)
        self.policy_out = nn.Linear(cfg.width, 1)

        self.board = BoardCNN(
            cfg.width, cfg.cnn_channels, cfg.cnn_blocks, cfg.board_rows, cfg.board_cols
        )
        self.cnn_token_in = nn.Linear(cfg.token_features, cfg.width)
        self.cnn_token_norm = RMSNorm(cfg.width)
        self.cnn_state_fuse = nn.Sequential(
            nn.Linear(cfg.width * 2, cfg.width), nn.SiLU(), nn.Linear(cfg.width, cfg.width)
        )
        self.cnn_state_norm = RMSNorm(cfg.width)
        self.cnn_action_in = nn.Linear(cfg.action_features, cfg.width)
        self.cnn_action_state = nn.Linear(cfg.width, cfg.width, bias=False)
        self.cnn_policy_hidden = nn.Linear(cfg.width, cfg.width)
        self.cnn_policy_norm = RMSNorm(cfg.width)
        self.cnn_policy_out = nn.Linear(cfg.width, 1)

        self.value_head = nn.Sequential(
            nn.Linear(cfg.width, cfg.width // 2), nn.SiLU(), nn.Linear(cfg.width // 2, 3)
        )
        self.aux_head = nn.Sequential(
            nn.Linear(cfg.width, cfg.width // 2),
            nn.SiLU(),
            nn.Linear(cfg.width // 2, cfg.aux_targets),
        )

    def forward(self, tokens, token_mask, actions, action_mask):
        pad = token_mask < 0.5
        x = self.token_in(tokens)
        for block in self.blocks:
            x = block(x, pad)
        x = self.norm(x)
        query = self.action_in(actions)
        attended, _ = self.policy_attn(query, x, x, key_padding_mask=pad, need_weights=False)
        transformer_logits = self.policy_out(
            self.policy_norm(query + attended)
        ).squeeze(-1)

        board = self.board(tokens, token_mask)
        token_h = self.cnn_token_norm(self.cnn_token_in(tokens))
        weight = token_mask.unsqueeze(-1)
        pooled = (token_h * weight).sum(dim=1) / weight.sum(dim=1).clamp(min=1.0)
        state = self.cnn_state_norm(
            self.cnn_state_fuse(torch.cat([board, pooled], dim=-1))
        )
        cnn_query = self.cnn_action_in(actions)
        conditioned = (
            cnn_query + self.cnn_action_state(state).unsqueeze(1)
            + cnn_query * state.unsqueeze(1)
        )
        cnn_hidden = self.cnn_policy_norm(
            self.cnn_policy_hidden(F.silu(conditioned))
        )
        cnn_logits = self.cnn_policy_out(cnn_hidden).squeeze(-1)

        logits = 0.5 * (transformer_logits + cnn_logits)
        logits = logits.masked_fill(action_mask < 0.5, float("-inf"))
        return logits, self.value_head(state), self.aux_head(state)

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


class SplitHeadCNNTransformer(nn.Module):
    """Transformer policy with a separate CNN-based value/auxiliary encoder.

    The policy path is structurally identical to TetraFormer-S and receives only
    policy gradients.  Value and auxiliary prediction use a compact board CNN
    fused with a masked mean of raw token embeddings, so local geometry can be
    learned without perturbing the policy representation through multitask loss.
    """

    architecture = "split_hybrid"

    def __init__(self, cfg: SplitHybridConfig):
        super().__init__()
        self.cfg = cfg
        tf_cfg = TetraFormerConfig(
            token_features=cfg.token_features,
            action_features=cfg.action_features,
            width=cfg.width,
            layers=cfg.layers,
            heads=cfg.heads,
            ffn=cfg.ffn,
            aux_targets=cfg.aux_targets,
            dropout=cfg.dropout,
        )

        # Policy path: same structure as the current TetraFormer-S control.
        self.token_in = nn.Linear(cfg.token_features, cfg.width)
        self.blocks = nn.ModuleList([EncoderBlock(tf_cfg) for _ in range(cfg.layers)])
        self.norm = RMSNorm(cfg.width)
        self.action_in = nn.Linear(cfg.action_features, cfg.width)
        self.policy_attn = nn.MultiheadAttention(
            cfg.width, cfg.heads, dropout=cfg.dropout, batch_first=True
        )
        self.policy_norm = RMSNorm(cfg.width)
        self.policy_out = nn.Linear(cfg.width, 1)

        # Independent value/aux path.  No parameter here is shared with policy.
        self.value_board = BoardCNN(
            cfg.width, cfg.cnn_channels, cfg.cnn_blocks, cfg.board_rows, cfg.board_cols
        )
        self.value_token_in = nn.Linear(cfg.token_features, cfg.width)
        self.value_token_norm = RMSNorm(cfg.width)
        self.value_fuse = nn.Sequential(
            nn.Linear(cfg.width * 2, cfg.width), nn.SiLU(), nn.Linear(cfg.width, cfg.width)
        )
        self.value_state_norm = RMSNorm(cfg.width)
        self.value_head = nn.Sequential(
            nn.Linear(cfg.width, cfg.width // 2), nn.SiLU(), nn.Linear(cfg.width // 2, 3)
        )
        self.aux_head = nn.Sequential(
            nn.Linear(cfg.width, cfg.width // 2),
            nn.SiLU(),
            nn.Linear(cfg.width // 2, cfg.aux_targets),
        )

    def forward(self, tokens, token_mask, actions, action_mask):
        pad = token_mask < 0.5
        x = self.token_in(tokens)
        for block in self.blocks:
            x = block(x, pad)
        x = self.norm(x)

        query = self.action_in(actions)
        attended, _ = self.policy_attn(query, x, x, key_padding_mask=pad, need_weights=False)
        logits = self.policy_out(self.policy_norm(query + attended)).squeeze(-1)
        logits = logits.masked_fill(action_mask < 0.5, float("-inf"))

        board = self.value_board(tokens, token_mask)
        token_h = self.value_token_norm(self.value_token_in(tokens))
        weight = token_mask.unsqueeze(-1)
        pooled = (token_h * weight).sum(dim=1) / weight.sum(dim=1).clamp(min=1.0)
        value_state = self.value_state_norm(
            self.value_fuse(torch.cat([board, pooled], dim=-1))
        )
        return logits, self.value_head(value_state), self.aux_head(value_state)

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


def build_ablation_model(architecture: str, token_features: int, action_features: int,
                         aux_targets: int) -> nn.Module:
    if architecture == "transformer":
        return TetraFormer(TetraFormerConfig(
            token_features=token_features,
            action_features=action_features,
            width=256,
            layers=8,
            heads=8,
            ffn=768,
            aux_targets=aux_targets,
        ))
    if architecture == "cnn":
        return CNNPolicyValue(CNNConfig(
            token_features=token_features,
            action_features=action_features,
            aux_targets=aux_targets,
        ))
    if architecture == "hybrid":
        return CNNTransformerHybrid(HybridConfig(
            token_features=token_features,
            action_features=action_features,
            aux_targets=aux_targets,
        ))
    if architecture == "split_hybrid":
        return SplitHeadCNNTransformer(SplitHybridConfig(
            token_features=token_features,
            action_features=action_features,
            aux_targets=aux_targets,
        ))
    if architecture == "fusion_hybrid":
        return FusionCNNTransformer(HybridConfig(
            token_features=token_features,
            action_features=action_features,
            aux_targets=aux_targets,
        ))
    if architecture == "dual_policy_hybrid":
        return DualPolicyCNNTransformer(HybridConfig(
            token_features=token_features,
            action_features=action_features,
            aux_targets=aux_targets,
        ))
    raise ValueError(f"unknown architecture: {architecture}")


def checkpoint_config(model: nn.Module) -> dict:
    cfg = getattr(model, "cfg", None)
    if cfg is None:
        raise ValueError("model has no serializable cfg")
    if hasattr(cfg, "__dataclass_fields__"):
        return asdict(cfg)
    return dict(cfg.__dict__)


def load_ablation_checkpoint(path: str, device: torch.device | str = "cpu") -> nn.Module:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(checkpoint, nn.Module):
        model = checkpoint
    else:
        architecture = checkpoint.get("architecture", "transformer")
        config = checkpoint["config"]
        if architecture == "transformer":
            model = TetraFormer(TetraFormerConfig(**config))
        elif architecture == "cnn":
            model = CNNPolicyValue(CNNConfig(**config))
        elif architecture == "hybrid":
            model = CNNTransformerHybrid(HybridConfig(**config))
        elif architecture == "split_hybrid":
            model = SplitHeadCNNTransformer(SplitHybridConfig(**config))
        elif architecture == "fusion_hybrid":
            model = FusionCNNTransformer(HybridConfig(**config))
        elif architecture == "dual_policy_hybrid":
            model = DualPolicyCNNTransformer(HybridConfig(**config))
        else:
            raise ValueError(f"unsupported checkpoint architecture: {architecture}")
        model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()
    return model
