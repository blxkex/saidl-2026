from typing import Literal

import torch as t
import torch.nn as nn
import torch.nn.functional as F

from attention_heads import FlexibleAttentionBlock, MaskedMultiHeadedAttention


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
        dim: int,
        attention: nn.Module,
    ):
        super().__init__()

        self.layer_norm1 = nn.LayerNorm(dim)
        self.layer_norm2 = nn.LayerNorm(dim)
        self.attention = attention

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


# padding issue fix (to prevent information leak from future tokens).
class CausalConv1d(nn.Conv1d):
    def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
        kwargs.pop('padding', None) # Force padding to 0
        super().__init__(in_channels, out_channels, kernel_size, padding=0, **kwargs)
        self.left_padding = kernel_size - 1

    def forward(self, x):
        x = F.pad(x, (self.left_padding, 0)) # Pad only the past
        return super().forward(x)
        

class ConvAttentionBlock(nn.Module):
    def __init__(
        self,
        seq_len: int,
        dim: int,
        kernel_size: int,
        padding: int,
        attention_block: nn.Module,
    ):
        super().__init__()

        self.conv1 = CausalConv1d(dim, dim, kernel_size=kernel_size)
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

        self.conv = CausalConv1d(dim, dim, kernel_size=kernel_size, padding=padding)
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

        residue = x

        x = self.norm2(x)
        x = self.mlp(x)

        x = x + residue

        return x


class ModularTransformer(nn.Module):
    """
    Modes:
      - "baseline":    n_layers x TransformerBlock
      - "hybrid":      n_layers x ConvAttentionBlock (Conv + Attention fused)
      - "alternating": ConvBlock, TransformerBlock, ConvBlock, ... (n_layers total)

    attention_blocks: list of pre-built attention modules. Caller (training code) builds them.
        baseline    -> len == n_layers
        hybrid      -> len == n_layers
        alternating -> len == n_layers // 2  (one per TransformerBlock at odd indices)
    conv_cfg:   {"kernel_size": int, "padding": int}. Required for hybrid/alternating.
    use_abs_pe: learned absolute PE on input. Disable when attention already has RoPE/ALiBi/RPE.
    """

    def __init__(
        self,
        ctx_len: int,
        dim: int,
        n_layers: int,
        vocab_size: int,
        attention_blocks: list[nn.Module],
        mode: Literal["baseline", "hybrid", "alternating"] = "baseline",
        conv_cfg: dict | None = None,
        use_abs_pe: bool = True,
    ):
        super().__init__()
        assert mode in ("baseline", "hybrid", "alternating"), f"unknown mode {mode}"

        self.mode = mode
        self.use_abs_pe = use_abs_pe

        self.token_emb = nn.Embedding(vocab_size, dim)
        if use_abs_pe:
            self.PE = nn.Embedding(ctx_len, dim)

        if mode == "baseline":
            assert len(attention_blocks) == n_layers, (
                f"baseline needs {n_layers} attention blocks, got {len(attention_blocks)}"
            )
            self.blocks = nn.ModuleList(
                [TransformerBlock(dim, attn) for attn in attention_blocks]
            )
        elif mode == "hybrid":
            assert conv_cfg is not None, "hybrid mode needs conv_cfg"
            assert len(attention_blocks) == n_layers, (
                f"hybrid needs {n_layers} attention blocks, got {len(attention_blocks)}"
            )
            self.blocks = nn.ModuleList(
                [
                    ConvAttentionBlock(
                        seq_len=ctx_len,
                        dim=dim,
                        kernel_size=conv_cfg["kernel_size"],
                        padding=conv_cfg["padding"],
                        attention_block=attn,
                    )
                    for attn in attention_blocks
                ]
            )
        else:  # alternating
            assert conv_cfg is not None, "alternating mode needs conv_cfg"
            n_attn = n_layers // 2
            assert len(attention_blocks) == n_attn, (
                f"alternating needs {n_attn} attention blocks, got {len(attention_blocks)}"
            )
            blocks = []
            attn_iter = iter(attention_blocks)
            for i in range(n_layers):
                if i % 2 == 0:
                    blocks.append(
                        ConvBlock(
                            dim=dim,
                            kernel_size=conv_cfg["kernel_size"],
                            padding=conv_cfg["padding"],
                        )
                    )
                else:
                    blocks.append(TransformerBlock(dim, next(attn_iter)))
            self.blocks = nn.ModuleList(blocks)

        self.ln_final = nn.LayerNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size)

    def forward(self, x):  # x: (B, L) token ids
        h = self.token_emb(x)
        if self.use_abs_pe:
            L = x.size(1)
            if L <= self.PE.num_embeddings:
                positions = t.arange(L, device=x.device)
                pe = self.PE(positions)
            else:
                # extrapolation: ViT-style positional interpolation of the learned
                # table up to length L (exact when L <= ctx_len).
                w = self.PE.weight.unsqueeze(0).permute(0, 2, 1)  # (1, dim, ctx_len)
                w = F.interpolate(w, size=L, mode="linear", align_corners=False)
                pe = w.permute(0, 2, 1).squeeze(0)  # (L, dim)
            h = h + pe
        for block in self.blocks:
            h = block(h)
        h = self.ln_final(h)
        return self.lm_head(h)  # (B, L, vocab_size) raw logits
