import torch as t
import torch.nn as nn
import torch.nn.functional as F

from attention_heads import MaskedMultiHeadedAttention


class MLP(nn.Module):
    def __init__(self, dim, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.fc1 = nn.Linear(dim, 4 * dim)
        self.fc2 = nn.Linear(4 * dim, dim)

    def forward(self, x):
        x = self.fc1(x)
        x = F.gelu(x)  # gaussian error linear unit.
        x = self.fc2(x)
        return x


class TransformerBlock(nn.Module):
    """
    Transformer block... Based on the architectural diagram from "Attention is all You Need".
    """

    def __init__(
        self,
        ctx_len: int,
        dim: int,
        heads: int = 8,
        attention: nn.Module = MaskedMultiHeadedAttention,
    ):
        super().__init__()

        self.layer_norm1 = nn.LayerNorm(dim)
        self.layer_norm2 = nn.LayerNorm(dim)
        self.attention = attention(heads, ctx_len, dim)

        self.mlp = MLP(dim)

    def forward(self, x):
        residue = x

        x = self.layer_norm1(x)
        x = self.attention(x)

        x += residue

        residue = x

        x = self.layer_norm2(x)
        x = self.mlp(x)

        x += residue

        return x


class ConvAttentionBlock(nn.Module):
    def __init__(
        self,
        heads: int,
        seq_len: int,
        dim: int,
        kernel_size: int,
        padding: int,
        attention_block: nn.Module,
    ):
        super().__init__()

        self.conv1 = nn.Conv1d(dim, dim, kernel_size=kernel_size, padding=padding)
        self.norm1 = nn.LayerNorm(dim)
        self.attn = attention_block
        self.norm2 = nn.LayerNorm(dim)
        self.gelu = nn.GELU()

        self.mlp = MLP(dim)

    def forward(self, x):

        residue = x
        x = self.norm1(x)

        x = x.permute(0, 2, 1)  # (B, E, L)
        x = self.conv1(x)
        x = x.permute(0, 2, 1)  # (B, L, E)

        x = self.gelu(x)
        x = self.attn(x)

        x = x + residue

        residue = x

        x = self.norm2(x)

        x = self.mlp(x)
        x = x + residue

        return x


class ConvBlock(nn.Module):
    def __init__(self, dim: int, kernel_size: int, padding: int):
        super().__init__()

        self.conv = nn.Conv1d(dim, dim, kernel_size=kernel_size, padding=padding)
        self.norm1 = nn.LayerNorm(dim)
        self.gelu = nn.GELU()
        self.mlp = MLP(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x):
        residue = x
        x = self.norm1(x)

        x = x.permute(0, 2, 1)  # (B, E, L)
        x = self.conv(x)
        x = x.permute(0, 2, 1)  # (B, L, E)

        x = self.gelu(x)

        x = x + residue

        x = self.norm2(x)
        x = self.mlp(x)

        x = x + residue

        return x


class ModularTransformer(nn.Module):
    def __init__(self, ctx_len, dim, heads, n_layers, vocab_size, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.token_emb = nn.Embedding(vocab_size, dim)
        self.PE = nn.Embedding(ctx_len, dim)  # learned positional embedding
        self.blocks = nn.ModuleList(
            [TransformerBlock(ctx_len, dim, heads) for i in range(n_layers)]
        )
        self.ln_final = nn.LayerNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size)

    def forward(self, x):  # x: (B, L) token ids
        positions = t.arange(x.size(1), device=x.device)
        x = self.token_emb(x) + self.PE(positions)  # (B, L, dim)
        for block in self.blocks:
            x = block(x)
        x = self.ln_final(x)
        return self.lm_head(x)  # (B, L, vocab_size) raw logits
