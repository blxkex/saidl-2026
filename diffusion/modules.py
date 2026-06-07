import math
import torch as t
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from diffusers import AutoencoderKL

device = "cuda" if t.cuda.is_available() else "cpu"


class LandscapeDataset(Dataset):
    def __init__(self, root_dir: str, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.image_paths = list(self.root_dir.glob("*.jpg")) 

    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image
    

class Patchify(nn.Module):
    def __init__(self, patch_size: int, hidden_size: int, in_channels: int):
        super().__init__()

        self.projection = nn.Conv2d(in_channels, hidden_size, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.projection(x)  # [B, hidden_size, H/patch_size, W/patch_size]
        x = x.flatten(2)  # [B, hidden_size, num_patches]
        x = x.transpose(1, 2)  # [B, num_patches, hidden_size]
        return x
    

class TimeStepEmbedding(nn.Module):
    def __init__(self, hidden_size: int, freq_dim: int):
        super().__init__()
        
        self.freq_dim = freq_dim

        self.mlp = nn.Sequential(
            nn.Linear(freq_dim, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size)
        )

    def forward(self, ts):
        # ts is a 1D tensor of integers like: [120, 999, 45, ...]
        
        # 1. pure math: generate the frequencies
        half = self.freq_dim // 2
        freqs = t.exp(
            -math.log(10000) * t.arange(half, dtype=t.float32, device=ts.device) / half
        )
        args = ts[:, None].float() * freqs[None]
        
        # 2. create the barcode by concatenating sine and cosine
        embedding = t.cat([t.cos(args), t.sin(args)], dim=-1)
        
        # 3. pass the barcode through the MLP
        return self.mlp(embedding)
    

class LabelEmbedder(nn.Module):
    def __init__(self, num_classes: int, hidden_size: int):
        super().__init__()
        # creates a matrix of size [num_classes, hidden_size]
        self.embedding = nn.Embedding(num_classes, hidden_size)

    def forward(self, labels):
        # just looks up the row corresponding to the label integer
        return self.embedding(labels)
    

class DiTBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()
        
        self.norm1 = nn.LayerNorm(hidden_size)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_size)

        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Linear(hidden_size * 4, hidden_size)
        )

        self.adaLN_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size * 6, bias=True)
        )

        # THE ZERO TRICK: initialize the final linear layer to strictly 0
        nn.init.constant_(self.adaLN_layer[1].weight, 0)
        nn.init.constant_(self.adaLN_layer[1].bias, 0)
    
    def forward(self, x, c):
         
        adaLN_params = self.adaLN_layer(c)  # [B, hidden_size * 6]

        # unsqueeze to broadcast across the sequence length L. shape becomes [B, 1, 4608]
        adaLN_params = adaLN_params.unsqueeze(1)

        # 6 parameters of dimension [B, 1, hidden_size] each
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = adaLN_params.chunk(6, dim=2)

        # pre normalization with adaptive shift and scale
        x_mod_1 = self.norm1(x) * (1 + scale_msa) + shift_msa

        attn_out, _ = self.attn(x_mod_1, x_mod_1, x_mod_1) # self attention

        x = x + gate_msa * attn_out  # residual connection with gating

        # pre normalization with adaptive shift and scale
        x_mod_2 = self.norm2(x) * (1 + scale_mlp) + shift_mlp
        mlp_out = self.mlp(x_mod_2)

        x = x + gate_mlp * mlp_out  # residual connection with gating

        return x
    

class FrozenVAE(nn.Module):
    def __init__(self, pretrained_model_name: str, device: str):
        super().__init__()
        self.autoencoder = AutoencoderKL.from_pretrained(pretrained_model_name).to(device)
        self.autoencoder.eval()
        
        # freeze weights
        for param in self.autoencoder.parameters():
            param.requires_grad = False

    def encode(self, x):
        with t.no_grad():
            z = self.autoencoder.encode(x).latent_dist
            latents = z.sample()
            return latents * 0.18215 # magic scaling factor
        

class DifficultyPredictor(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.SiLU(),
            nn.Linear(256, 1),
            nn.Sigmoid() # forces the output to be strictly between [0.0, 1.0]
        )

    
    def forward(self, dit_features):
        # dit_features shape: [B, 16, 768]
        # output shape: [B, 16, 1] (one difficulty score per patch)
        return self.net(dit_features)