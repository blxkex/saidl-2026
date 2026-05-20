import math

import torch as t
import torch.nn as nn
import torch.nn.functional as F

from typing import Literal
from positional_embeddings import *


class SelfAttention(nn.Module):
    """
    (B, L, E) -> (B, L, E)
    """

    def __init__(self, ctx_len, dim, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.dim = dim
        self.Qw = nn.Linear(dim, dim)
        self.Kw = nn.Linear(dim, dim)
        self.Vw = nn.Linear(dim, dim)

    def forward(self, x, mask=None):

        # (B, L, E) -> (B, L, E)

        Q = self.Qw(x)
        K = self.Kw(x)
        V = self.Vw(x)

        dot_prod = Q @ K.permute(0, 2, 1)  # (B, L, E) x (B, E, L) = (B, L , L)

        # size = (B, L, L)
        scaled_dot = dot_prod / math.sqrt(self.dim)

        # 3. Apply the Mask
        if mask is not None:
            # .masked_fill takes a condition, and replaces values with what you tell it.
            # Assuming mask has 0s where we want to hide stuff, and 1s where it's safe to look.
            masked_scaled_dot = scaled_dot.masked_fill(mask == 0, float("-inf"))

        attention_weights = F.softmax(masked_scaled_dot, dim=-1)

        # (B, L, L) x (B, L, E) = (B, L, E)
        out = attention_weights @ V

        return out


class MaskedMultiHeadedAttention(nn.Module):
    """
    (B, L, E) -> (B, n_heads, L, E/n_heads) [goes through multiple attention heads by splitting up the embedding dimensions] -> (B, L, E)
    """

    def __init__(self, heads, ctx_len, dim, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert dim % heads == 0, "dim must be divisible by n_heads"

        self.dim = dim
        self.heads = heads  # no.of attention heads
        self.head_dim = dim // heads

        self.Qw = nn.Linear(dim, dim)
        self.Kw = nn.Linear(dim, dim)
        self.Vw = nn.Linear(dim, dim)
        self.Wo = nn.Linear(dim, dim)  # output projection.

        causal_mask = t.tril(t.ones(ctx_len, ctx_len))
        self.register_buffer("causal_mask", causal_mask)

    def forward(self, x: t.Tensor):

        B, L, _ = x.size()

        # (B, L, E) -> (B, L, n, E/n) -> (B, n, L, E/n)
        Q = self.Qw(x).view(B, L, self.heads, self.head_dim).permute(0, 2, 1, 3)
        K = self.Kw(x).view(B, L, self.heads, self.head_dim).permute(0, 2, 1, 3)
        V = self.Vw(x).view(B, L, self.heads, self.head_dim).permute(0, 2, 1, 3)

        dot_prod = Q @ K.permute(
            0, 1, 3, 2
        )  # (B, n, L, E/n) x (B, n, E/n, L) = (B, n, L, L)

        # size = (B, n, L, L)
        scaled_dot = dot_prod / math.sqrt(self.head_dim)

        # applying causal mask.
        causal = self.causal_mask[
            :L, :L
        ]  # slicing to the current length (only affects the edge cases).
        masked_scaled_dot = scaled_dot.masked_fill(causal == 0, float("-inf"))

        attention_weights = F.softmax(masked_scaled_dot, dim=-1)

        # (B, n, L, L) x (B, n, L, E/n) = (B, n, L, E/n)
        out = attention_weights @ V

        # we first permute it, and then join emb_dims from diff heads together.
        out = out.permute(0, 2, 1, 3).contiguous().view(B, L, self.dim)  # (B, L, E)
        out = self.Wo(out)

        return out


# Fits in all the variants (For PE variants and Attention Variants)
class FlexibleAttentionBlock(nn.Module):
    def __init__(
        self,
        pe: Literal["RoPE", "ALiBI", "RPE"],
        variant: Literal["SWA", "MQA", "GQA"],
        dim,
        seq_len,
        heads,
        # some specific arguments for the variants smh.
        window_size: int | None = None,
        max_distance: int | None = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        assert dim % heads == 0, "dim must be divisible by n_heads"

        self.pe = pe
        self.variant = variant

        if self.pe == "RoPE":
            self.PE = RoPE(dim=dim, seq_len=seq_len)

        elif self.pe == "ALiBI":
            self.PE = ALiBi(heads=heads, seq_len=seq_len)

        elif self.pe == "RPE":
            assert (
                max_distance is not None
            ), "Pass in max_distance if you want to use RPE"
            self.PE = RPE(heads=heads, seq_len=seq_len, max_distance=max_distance)

        if self.variant == "SWA":
            assert (
                window_size is not None
            ), "Nigga, pass in `window_size` if you using SWA."
            positions = t.arange(seq_len)

            # gives a matrix of (seq_len, seq_len). An anti-symmetric matrix.
            distances = positions.unsqueeze(1) - positions.unsqueeze(0)

            allowed = (distances >= 0) & (
                distances < window_size
            )  # gives a matrix filled with True/False.

            self.attn_mask = t.where(allowed, 0.0, float("-inf"))

        self.Qw = nn.Linear

        self.dim = dim
        self.heads = heads  # no.of attention heads
        self.head_dim = dim // heads

        self.Qw = nn.Linear(dim, dim)
        self.Kw = nn.Linear(dim, dim)
        self.Vw = nn.Linear(dim, dim)
        self.Wo = nn.Linear(dim, dim)  # output projection.

    def forward(self, x: t.Tensor):

        B, L, _ = x.size()

        PreQ = self.Qw(x)
        PreK = self.Kw(x)
        PreV = self.Vw(x)

        if self.pe == "RoPE":
            Q, K = self.PE(PreQ, PreK)

        Q = Q.view(B, L, self.heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(B, L, self.heads, self.head_dim).permute(0, 2, 1, 3)
        V = PreV.view(B, L, self.heads, self.head_dim).permute(0, 2, 1, 3)

        if self.variant == "SWA":

            attn_weights = Q @ K.permute(
                0, 1, 3, 2
            )  # (B, n, L, E/n) x (B, n, E/n, L) = (B, n, L, L)

            # size = (B, n, L, L)
            scaled_attn_weights = attn_weights / math.sqrt(self.head_dim)

            if self.pe == "ALiBi":
                scaled_attn_weights = self.PE(ALiBi)
            elif self.pe == "RPE":
                scaled_attn_weights = self.PE(ALiBi)

            masked_attn = scaled_attn_weights + self.attn_mask

            attn_scores = F.softmax(masked_attn, dim=-1)

            # final output
            out = attn_scores @ V  # (B, n, L, L) x (B, n, L, E/n) = (B, n, L, E/n)
            out = out.permute(0, 2, 1, 3).contiguous().view(B, L, self.dim)  # (B, L, E)
            out = self.Wo(out)

            return out
