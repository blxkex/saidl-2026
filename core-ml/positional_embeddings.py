import torch as t
import torch.nn as nn


# my naive implementation.
def baseline_pe(emb: t.Tensor, dim: int, pos: int):
    "adds positional embedding to the input vectors. Embeddings -> Embeddings + P.E"

    # defining the mathematical functions.
    def sin_pe(i, pos, dim):
        return t.sin(pos / 10000 ** (2 * i / dim))

    def cos_pe(i, pos, dim):
        return t.cos(pos / 10000 ** (2 * i / dim))

    for idx in range(dim):
        if idx % 2 == 0:
            emb[idx] += sin_pe(idx / 2, pos, dim)
        else:
            emb[idx] += cos_pe(idx // 2, pos, dim)

    return emb


# A better implementation. Never knew you could vectorize like this lmfao, but here we are.
class BaselinePE(nn.Module):
    def __init__(self, dim: int, len: int = 1024):
        super().__init__()
        self.dim = dim

        # Registering as a buffer so it saves with the model but doesn't get trained
        self.register_buffer("positional_encodings", self._build(len))

    def _build(self, length, device=None):
        # blank canvas
        positional_encodings = t.zeros(length, self.dim, device=device)

        # positions and indices
        positions = t.arange(0, length, dtype=t.float, device=device).unsqueeze(1)
        even_idx = t.arange(0, self.dim, 2, dtype=t.float, device=device)

        # pre-calculating the denominator and its fractional value.
        denominator = 10000 ** (even_idx / self.dim)
        inv_denom = (1 / denominator).unsqueeze(0)

        # This was honestly crazy. Slicing through at intervals of two so that it fills only even/odd rows.
        positional_encodings[:, 0::2] = t.sin(positions @ inv_denom)
        positional_encodings[:, 1::2] = t.cos(positions @ inv_denom)

        # Now since we have the (B, L, E) dimensions in mind.
        return positional_encodings.unsqueeze(0)

    def forward(self, x):
        L = x.size(1)
        # extrapolation: extend cache on-device if input is longer than cached
        if L > self.positional_encodings.size(1):
            self.positional_encodings = self._build(L, x.device)
        x += self.positional_encodings[:, :L]
        return x


class RoPE(nn.Module):
    def __init__(self, dim: int, seq_len: int):
        super().__init__()

        # theta = 10000^{-2i/d} part
        self.theta = 10000 ** (-2 * t.arange(0, dim // 2) / dim).unsqueeze(
            -1
        )  # shape = [128, 1]

        self.register_buffer("e", self._build(seq_len))

    def _build(self, seq_len, device=None):
        theta = self.theta.to(device) if device is not None else self.theta
        # matmul stays float; multiply by 1j only after.
        angles = (
            t.arange(seq_len, dtype=t.float, device=device).unsqueeze(-1) @ theta.T
        )  # shape = [seq_len, 1] x [1, 128] = [seq_len, 128].
        e = t.exp(1j * angles)
        return e.unsqueeze(0)  # [seq_len, 128] -> [1, seq_len, 128]

    def forward(self, xq, xk):

        # extrapolation: extend rotation cache on-device if input is longer than cached
        L = xq.size(-2)
        if L > self.e.size(1):
            self.e = self._build(L, xq.device)
        e = self.e[:, :L]

        # pairing them up along the embedding dimension: [1, 2, 3, 4] -> [[1, 2], [3, 4]]
        xq_paired = xq.reshape(*xq.shape[:-1], -1, 2)
        # OLD (wrong): used xq.shape for xk — breaks GQA/MQA where K has fewer heads.
        # xk_paired = xk.reshape(*xq.shape[:-1], -1, 2)
        # NEW: reshape xk using its own shape.
        xk_paired = xk.reshape(*xk.shape[:-1], -1, 2)

        # view them as complex: [[1, 2], [3, 4]] -> [[1, 2j], [3, 4j]]
        xq_comp = t.view_as_complex(xq_paired)
        xk_comp = t.view_as_complex(xk_paired)

        # rotato.
        xq_rot = xq_comp * e
        xk_rot = xk_comp * e

        # back to real
        xq_out = t.view_as_real(xq_rot).flatten(-2)
        xk_out = t.view_as_real(xk_rot).flatten(-2)

        return xq_out, xk_out


class ALiBi(nn.Module):
    def __init__(self, heads: int, seq_len: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.heads = heads
        self.seq_len = seq_len

        self.register_buffer("alibi_bias", self._build_bias(seq_len))
        # Create the causal mask here and register it as a buffer
        self.register_buffer("causal_mask", self._build_mask(seq_len))

    def _build_bias(self, seq_len, device=None):
        positions = t.arange(seq_len, dtype=t.float32, device=device)
        distances = positions.unsqueeze(1) - positions.unsqueeze(0)

        # negative so that the penalties remain negative.
        penals = -t.abs(distances)

        slopes = 2 ** -(t.arange(self.heads, dtype=t.float32, device=device) * 8 / self.heads)
        slopes = slopes.reshape(self.heads, 1, 1)
        return slopes * penals

    def _build_mask(self, seq_len, device=None):
        return t.triu(t.full((seq_len, seq_len), float("-inf"), device=device), diagonal=1)

    def forward(self, x):
        # extrapolation: extend bias + mask on-device if input is longer than cached
        L = x.size(-1)
        if L > self.causal_mask.size(-1):
            self.alibi_bias = self._build_bias(L, x.device)
            self.causal_mask = self._build_mask(L, x.device)
        # Now both buffers are automatically on the same device as the model (CUDA)
        final_attn_mask = self.alibi_bias[..., :L, :L] + self.causal_mask[:L, :L]

        return x + final_attn_mask


class RPE(nn.Module):
    def __init__(self, heads: int, seq_len: int, max_distance: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_distance = max_distance

        # shape = (distance buckets, number of heads)
        self.rpe_bias_table = nn.Embedding(max_distance + 1, heads)

        self.register_buffer("c_dist_cache", self._build(seq_len))

    def _build(self, seq_len, device=None):
        positions = t.arange(seq_len, device=device)
        distances = positions.unsqueeze(1) - positions.unsqueeze(0)  # shape = (L, L)

        return t.clamp(
            distances, min=0, max=self.max_distance
        )  # automatically makes the negative become zero.

    def forward(self, x):

        cur_len = x.size(-1)
        # extrapolation: extend distance cache on-device if input is longer than cached
        if cur_len > self.c_dist_cache.size(-1):
            self.c_dist_cache = self._build(cur_len, x.device)
        c_dist = self.c_dist_cache[:cur_len, :cur_len]

        # index the (n, buckets) table directly into (n, L, L); avoids the
        # (L, L, n) intermediate + permute copy -> lower peak memory at large L.
        rpe_bias = self.rpe_bias_table.weight.t()[:, c_dist].unsqueeze(0)  # (1, n, L, L)

        return x + rpe_bias
