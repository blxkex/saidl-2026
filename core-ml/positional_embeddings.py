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

        # blank canvas
        positional_encodings = t.zeros(len, dim)

        # positions and indices
        positions = t.arange(0, len, dtype=t.float).unsqueeze(1)
        even_idx = t.arange(0, dim, 2, dtype=t.float)

        # pre-calculating the denominator and its fractional value.
        denominator = 10000 ** (even_idx / dim)
        inv_denom = (1 / denominator).unsqueeze(0)

        # This was honestly crazy. Slicing through at intervals of two so that it fills only even/odd rows.
        positional_encodings[:, 0::2] = t.sin(positions @ inv_denom)
        positional_encodings[:, 1::2] = t.cos(positions @ inv_denom)

        # Now since we have the (B, L, E) dimensions in mind.
        positional_encodings = positional_encodings.unsqueeze(0)

        # Registering as a buffer so it saves with the model but doesn't get trained
        self.register_buffer("positional_encodings", positional_encodings)

    def forward(self, x):
        x += self.positional_encodings
        return x


class RoPE(nn.Module):
    def __init__(self, dim: int, seq_len: int):
        super().__init__()

        # theta = 10000^{-2i/d} part
        self.theta = 10000 ** (-2 * t.arange(0, dim // 2) / dim).unsqueeze(
            -1
        )  # shape = [128, 1]

        # converting to e^{i m theta}
        self.e = t.exp(
            1j * t.arange(seq_len).unsqueeze(-1) @ self.theta.T
        )  # shape = [1024, 1] x [1, 128] = [1024, 128].

        self.e = self.e.unsqueeze(0)  # [1024, 128] -> [1, 1024, 128]

        self.register_buffer("e", self.e)

    def forward(self, xq, xk):

        # pairing them up along the embedding dimension: [1, 2, 3, 4] -> [[1, 2], [3, 4]]
        xq_paired = xq.reshape(*xq.shape[:-1], -1, 2)
        xk_paired = xk.reshape(*xq.shape[:-1], -1, 2)

        # view them as complex: [[1, 2], [3, 4]] -> [[1, 2j], [3, 4j]]
        xq_comp = t.view_as_complex(xq_paired)
        xk_comp = t.view_as_complex(xk_paired)

        # rotato.
        xq_rot = xq_comp * self.e
        xk_rot = xk_comp * self.e

        # back to real
        xq_out = t.view_as_real(xq_rot).flatten(-2)
        xk_out = t.view_as_real(xk_rot).flatten(-2)

        return xq_out, xk_out


class ALiBi(nn.Module):
    def __init__(self, heads: int, seq_len: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.seq_len = seq_len

        positions = t.arange(seq_len, dtype=t.float32)
        distances = positions.unsqueeze(1) - positions.unsqueeze(0)

        # negative so that the penalties remain negative.
        penals = -t.abs(distances)

        slopes = 2 ** -(t.arange(heads, dtype=t.float32) * 8 / heads)
        slopes = slopes.reshape(heads, 1, 1)

        self.register_buffer("alibi_bias", slopes * penals)

    def forward(self, x):
        causal_mask = t.triu(
            t.full((self.seq_len, self.seq_len), float("-inf")), diagonal=1
        )
        final_attn_mask = self.alibi_bias + causal_mask

        return x + final_attn_mask


class RPE(nn.Module):
    def __init__(self, heads: int, seq_len: int, max_distance: int, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # shape = (distance buckets, number of heads)
        self.rpe_bias_table = nn.Embedding(max_distance + 1, heads)

        positions = t.arange(seq_len)
        distances = positions.unsqueeze(1) - positions.unsqueeze(0)  # shape = (L, L)

        c_dist = t.clamp(
            distances, min=0, max=max_distance
        )  # automatically makes the negative become zero.

        self.register_buffer("c_dist_cache", c_dist)

    def forward(self, x):

        cur_len = x.size(-1)
        c_dist = self.c_dist_cache[:cur_len, :cur_len]

        b: t.Tensor = self.rpe_bias_table(c_dist)  # dim: (L, L, n)

        rpe_bias = b.permute(2, 0, 1).unsqueeze(0)  # dim: (1, n, L, L)

        return x + rpe_bias
