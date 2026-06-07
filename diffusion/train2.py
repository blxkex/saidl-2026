import os
# keep CUDA fragmentation down on the 6GB card (set before torch loads)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from modules import *
from DiT import DiT
from torch.optim import Adam

import wandb

ROOT = Path(__file__).resolve().parent
DIT_CKPT = ROOT / "checkpoints" / "dit_best.pt"
PREDICTOR_CKPT = ROOT / "checkpoints" / "difficulty_predictor.pt"
LATENT_CACHE = ROOT / "latent_cache.pt"

DIT_KWARGS = dict(hidden_size=768, patch_size=8, num_heads=12,
                  num_blocks=12, num_classes=1, in_channels=4)

NUM_TIMESTEPS = 1000
EPOCHS = 10
BATCH_SIZE = 32
PATCH = 8


def get_noise_schedule(num_timesteps=NUM_TIMESTEPS, device="cuda"):
    beta = t.linspace(0.0001, 0.02, num_timesteps, device=device)
    alpha = 1.0 - beta
    return t.cumprod(alpha, dim=0)


def q_sample(x_start, t_indices, noise, alpha_bar):
    a_bar = alpha_bar[t_indices].view(-1, 1, 1, 1)
    return t.sqrt(a_bar) * x_start + t.sqrt(1 - a_bar) * noise


def patchify(x, patch=PATCH):
    # [B, 4, 32, 32] -> [B, 16, 4*8*8] : raw pixel patches matching the DiT grid
    B, C, H, W = x.shape
    x = x.unfold(2, patch, patch).unfold(3, patch, patch)      # [B, C, 4, 4, 8, 8]
    x = x.permute(0, 2, 3, 1, 4, 5).reshape(B, (H // patch) * (W // patch), C * patch * patch)
    return x


def train_predictor():
    device = "cuda" if t.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    wandb.init(project="saidl-diffusion", name="difficulty-predictor",
               entity="bhuvaneshreddy-bits-pilani",
               config={"epochs": EPOCHS, "batch_size": BATCH_SIZE, "lr": 1e-3})

    # 1. Frozen DiT
    dit_model = DiT(**DIT_KWARGS).to(device)
    dit_model.load_state_dict(t.load(DIT_CKPT, map_location=device))
    dit_model.eval()
    for param in dit_model.parameters():
        param.requires_grad = False

    # 2. Data = cached clean latents (the baseline's encode pass)
    latents = t.load(LATENT_CACHE)["latents"]  # [N, 4, 32, 32], fp16, CPU
    loader = DataLoader(t.utils.data.TensorDataset(latents),
                        batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    alpha_bar = get_noise_schedule(NUM_TIMESTEPS, device)

    # 3. Predictor + optimizer
    predictor = DifficultyPredictor(hidden_size=DIT_KWARGS["hidden_size"]).to(device)
    predictor_optimizer = Adam(predictor.parameters(), lr=1e-3)

    # 4. Phase 2 loop: predict per-patch difficulty from frozen DiT features
    predictor.train()
    for epoch in range(EPOCHS):
        epoch_loss = 0.0
        for (x_0,) in loader:
            x_0 = x_0.to(device).float()
            B = x_0.shape[0]

            ts = t.randint(0, NUM_TIMESTEPS, (B,), device=device, dtype=t.long)
            y = t.zeros(B, dtype=t.long, device=device)
            noise = t.randn_like(x_0)
            x_t = q_sample(x_0, ts, noise, alpha_bar)

            with t.no_grad():
                predicted_noise, features = dit_model.forward_with_features(x_t, ts, y)
                # predicted x_0 via Tweedie's formula
                a_bar = alpha_bar[ts].view(-1, 1, 1, 1)
                predicted_x0 = (x_t - t.sqrt(1 - a_bar) * predicted_noise) / t.sqrt(a_bar)

            # true per-patch difficulty = MSE between predicted and clean x_0
            true_difficulty = F.mse_loss(patchify(predicted_x0), patchify(x_0),
                                         reduction="none").mean(dim=-1, keepdim=True)  # [B, 16, 1]
            true_difficulty = t.clamp(true_difficulty / true_difficulty.max(), 0, 1)

            # train predictor to guess it
            predicted_difficulty = predictor(features)  # [B, 16, 1]
            loss = F.mse_loss(predicted_difficulty, true_difficulty)

            predictor_optimizer.zero_grad()
            loss.backward()
            predictor_optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(loader)
        wandb.log({"predictor/loss": avg_loss, "epoch": epoch})
        print(f"Epoch {epoch} | Predictor MSE: {avg_loss:.5f}")

    # 5. Save
    PREDICTOR_CKPT.parent.mkdir(exist_ok=True)
    t.save(predictor.state_dict(), PREDICTOR_CKPT)
    print(f"Saved difficulty predictor to {PREDICTOR_CKPT}")
    wandb.finish()


if __name__ == "__main__":
    train_predictor()
