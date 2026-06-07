from modules import *

class DiT(nn.Module):
    def __init__(self, num_blocks: int, hidden_size: int, num_heads: int, num_classes: int, patch_size: int, in_channels: int, input_size: int = 32):
        super().__init__()

        self.patch_size = patch_size
        self.in_channels = in_channels

        self.time_embedder = TimeStepEmbedding(hidden_size, freq_dim=256)
        self.label_embedder = LabelEmbedder(num_classes, hidden_size)
        self.patchify = Patchify(patch_size, hidden_size, in_channels)

        # learnable positional embedding for each patch token. without this the
        # attention is permutation-invariant and has no idea where each patch
        # sits in the grid -> spatially incoherent ("patchy") samples.
        num_patches = (input_size // patch_size) ** 2
        self.pos_embed = nn.Parameter(t.zeros(1, num_patches, hidden_size))
        nn.init.normal_(self.pos_embed, std=0.02)

        self.dit_blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads) for _ in range(num_blocks)
        ])

        self.final_norm = nn.LayerNorm(hidden_size)
        self.final_linear = nn.Linear(hidden_size, patch_size * patch_size * in_channels)

    def forward(self, x, ts, labels):
        # x: [B, num_patches, hidden_size]
        # ts: [B]
        # labels: [B]
        noise, _ = self.forward_with_features(x, ts, labels)
        return noise

    def forward_with_features(self, x, ts, labels):
        # same as forward, but also returns the last DiT block's patch-token
        # features [B, num_patches, hidden_size] for the difficulty predictor

        x = self.patchify(x)
        x = x + self.pos_embed  # tell each token where it is in the patch grid

        time_emb = self.time_embedder(ts)  # [B, hidden_size]
        label_emb = self.label_embedder(labels)  # [B, hidden_size]

        # add the time and label embeddings to each patch token
        x = x + time_emb[:, None, :] + label_emb[:, None, :]   # replacement for unsqueeze, results in broadcasting.

        cond = time_emb + label_emb
        for block in self.dit_blocks:
            x = block(x, cond)

        features = x  # [B, num_patches, hidden_size]

        x = self.final_norm(x)
        x = self.final_linear(x)  # [B, num_patches, patch_size * patch_size * in_channels]

        x = self.unpatchify(x)  # [B, in_channels, H, W]

        return x, features
    
    def unpatchify(self, x):

        B, num_patches, _ = x.shape
        grid_size = int(math.sqrt(num_patches)) 
        
        # reshape [B, 16, 256] -> [B, 4, 4, 4, 8, 8]
        x = x.reshape(B, grid_size, grid_size, self.in_channels, self.patch_size, self.patch_size)
        
        # reorder dimensions to [B, 4, 4, 8, 4, 8] and flatten into [B, 4, 32, 32]
        x = x.permute(0, 3, 1, 4, 2, 5).contiguous()

        x = x.view(B, self.in_channels, grid_size * self.patch_size, grid_size * self.patch_size)
        
        return x

        