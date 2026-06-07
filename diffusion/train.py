import os
# reduce CUDA fragmentation on the 6GB card (must be set before torch loads)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import time

from modules import *
from DiT import DiT
from torch.optim import AdamW
from torch.amp import autocast, GradScaler
from torchvision.utils import save_image

import wandb
from torchmetrics.image.fid import FrechetInceptionDistance
from transformers import CLIPProcessor, CLIPVisionModelWithProjection

# how often (in epochs) to run the FID / CLIP evaluation
EVAL_EVERY = 5
# number of images to sample for the metrics
NUM_EVAL_SAMPLES = 200
# DDIM steps used when sampling for evaluation (full 1000 is too slow to do often)
EVAL_DDIM_STEPS = 50

ROOT = Path(__file__).resolve().parent
CHECKPOINT_DIR = ROOT / "checkpoints"
SAMPLE_DIR = ROOT / "samples"
# pre-encoded VAE latents live here so we only pay the VAE cost once, ever
LATENT_CACHE = ROOT / "latent_cache.pt"


def get_noise_schedule(num_timesteps=1000, device="cuda"):
    # standard linear schedule used in DDPM
    beta = t.linspace(0.0001, 0.02, num_timesteps, device=device)
    alpha = 1.0 - beta
    alpha_bar = t.cumprod(alpha, dim=0)
    return alpha_bar

def q_sample(x_start, t_indices, noise, alpha_bar):
    # grab the exact alpha_bar value for the random timesteps in the batch
    a_bar = alpha_bar[t_indices]
    expand_shape = [x_start.shape[0]] + [1] * (x_start.ndim - 1)
    a_bar = a_bar.view(*expand_shape)

    # math: sqrt(a_bar)*x_0 + sqrt(1 - a_bar)*noise
    noisy_latents = t.sqrt(a_bar) * x_start + t.sqrt(1 - a_bar) * noise
    return noisy_latents


def build_latent_cache(device):
    # Encode the whole dataset into VAE latents a single time and store them.
    # The transform is deterministic (no random augmentation), so the latents
    # never change between epochs -> safe to cache and reuse.
    if LATENT_CACHE.exists():
        print(f"Loading cached latents from {LATENT_CACHE}")
        return t.load(LATENT_CACHE)

    print("No cache found. Encoding dataset latents (one-time cost)...")
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]) # maps [0, 1] to [-1, 1]
    ])

    dataset_dir = ROOT / "dataset"
    dataset = LandscapeDataset(root_dir=dataset_dir, transform=transform)
    if len(dataset) == 0:
        raise RuntimeError(f"No dataset images found at {dataset_dir}")

    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)
    vae = FrozenVAE("stabilityai/sd-vae-ft-ema", device)

    all_latents = []
    real_pixels = []  # keep a few real images as the reference set for FID/CLIP
    with t.no_grad():
        for pixels in loader:
            pixels = pixels.to(device)
            with autocast("cuda"):
                latents = vae.encode(pixels)
            all_latents.append(latents.float().half().cpu())  # fp16 -> small file
            if sum(p.shape[0] for p in real_pixels) < NUM_EVAL_SAMPLES:
                real_pixels.append(pixels.cpu())

    cache = {
        "latents": t.cat(all_latents, dim=0),
        "real_pixels": t.cat(real_pixels, dim=0)[:NUM_EVAL_SAMPLES],
    }
    t.save(cache, LATENT_CACHE)
    print(f"Cached {cache['latents'].shape[0]} latents to {LATENT_CACHE}")

    del vae
    t.cuda.empty_cache()
    return cache


@t.no_grad()
def ddim_sample(model, alpha_bar, num_samples, device, num_steps=EVAL_DDIM_STEPS):
    # deterministic DDIM sampler (eta=0). model predicts the noise epsilon.
    model.eval()

    # walk timesteps backwards, e.g. 999 -> ... -> 0
    step_indices = t.linspace(999, 0, num_steps + 1, device=device).long()

    latents = []
    for start in range(0, num_samples, 16):
        b = min(16, num_samples - start)
        x = t.randn(b, 4, 32, 32, device=device)
        y = t.zeros(b, dtype=t.long, device=device)

        for i in range(num_steps):
            ti, ti_next = step_indices[i], step_indices[i + 1]
            ts = t.full((b,), ti, device=device, dtype=t.long)

            with autocast("cuda"):
                eps = model(x, ts, y).float()

            a_bar = alpha_bar[ti]
            a_bar_next = alpha_bar[ti_next]

            # predict x_0, then step to the next (less noisy) latent
            x0 = (x - t.sqrt(1 - a_bar) * eps) / t.sqrt(a_bar)
            x0 = x0.clamp(-1, 1)
            x = t.sqrt(a_bar_next) * x0 + t.sqrt(1 - a_bar_next) * eps

        latents.append(x)

    model.train()
    return t.cat(latents, dim=0)


@t.no_grad()
def decode_latents(vae, latents):
    # latents -> pixels in [-1, 1] using the frozen VAE decoder.
    # Decoding at 256x256 is memory-heavy, so we go in small chunks and move the
    # result to CPU immediately to keep VRAM flat (matters on the 6GB card).
    images = []
    for start in range(0, latents.shape[0], 8):
        z = latents[start:start + 8] / 0.18215  # undo the encode scaling
        with autocast("cuda"):
            decoded = vae.autoencoder.decode(z).sample.float()
        images.append(decoded.cpu())
    return t.cat(images, dim=0)


def to_uint8(images):
    # [-1, 1] float -> [0, 255] uint8, ready for the metric models
    images = (images.clamp(-1, 1) + 1) / 2
    return (images * 255).round().to(t.uint8)


def compute_cmmd(real_feats, fake_feats, sigma=10.0, scale=1000.0):
    # CMMD = Maximum Mean Discrepancy between CLIP embeddings with a Gaussian
    # (RBF) kernel. From "Rethinking FID" (Jayasumana et al., 2023). Unlike FID
    # it assumes no Gaussian distribution and stays reliable with few samples.
    # Embeddings are L2-normalized first; sigma/scale follow the official impl.
    x = F.normalize(real_feats.double(), dim=-1)
    y = F.normalize(fake_feats.double(), dim=-1)

    gamma = 1.0 / (2 * sigma ** 2)
    x_sq = (x * x).sum(dim=1)
    y_sq = (y * y).sum(dim=1)

    # squared euclidean distances -> Gaussian kernel matrices
    k_xx = t.exp(-gamma * (x_sq[:, None] + x_sq[None, :] - 2 * (x @ x.t())))
    k_yy = t.exp(-gamma * (y_sq[:, None] + y_sq[None, :] - 2 * (y @ y.t())))
    k_xy = t.exp(-gamma * (x_sq[:, None] + y_sq[None, :] - 2 * (x @ y.t())))

    mmd = k_xx.mean() + k_yy.mean() - 2 * k_xy.mean()
    return float((scale * mmd).clamp(min=0).item())


@t.no_grad()
def compute_metrics(real_u8, fake_u8, device):
    # FID (Inception + Frechet) and CMMD (CLIP + MMD) between two image sets,
    # both given as uint8 [N,3,H,W] tensors. Heavy metric models are loaded here
    # and freed right after to fit the 6GB card. Shared by train.py and generate.py.
    fid = FrechetInceptionDistance(feature=2048, normalize=False).to(device)
    for start in range(0, real_u8.shape[0], 16):
        fid.update(real_u8[start:start + 16].to(device), real=True)
    for start in range(0, fake_u8.shape[0], 16):
        fid.update(fake_u8[start:start + 16].to(device), real=False)
    fid_score = float(fid.compute().item())
    del fid
    t.cuda.empty_cache()

    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    clip_model = CLIPVisionModelWithProjection.from_pretrained(
        "openai/clip-vit-base-patch32"
    ).to(device).eval()

    def clip_features(images_u8):
        feats = []
        for start in range(0, images_u8.shape[0], 16):
            batch = list(images_u8[start:start + 16].cpu())  # CHW uint8 tensors
            inputs = clip_processor(images=batch, return_tensors="pt").to(device)
            feats.append(clip_model(**inputs).image_embeds)
        return t.cat(feats, dim=0)

    cmmd = compute_cmmd(clip_features(real_u8), clip_features(fake_u8))
    del clip_model, clip_processor
    t.cuda.empty_cache()
    return fid_score, cmmd


@t.no_grad()
def evaluate(model, vae, alpha_bar, real_images, device, epoch):
    # real_images: a fixed batch of real pixels in [-1, 1] used as the reference
    print(f"Running evaluation at epoch {epoch}...")
    t.cuda.empty_cache()  # release cached training blocks before the eval models load

    # 1. sample fake images, timing the generation for the compute comparison
    # (baseline vs cyclic vs RACD are judged on FID/CMMD *and* generation time)
    if device == "cuda":
        t.cuda.synchronize()
    gen_start = time.time()
    fake_latents = ddim_sample(model, alpha_bar, NUM_EVAL_SAMPLES, device)
    if device == "cuda":
        t.cuda.synchronize()
    gen_time = time.time() - gen_start
    gen_time_per_image = gen_time / NUM_EVAL_SAMPLES

    fake_pixels = decode_latents(vae, fake_latents)
    del fake_latents
    t.cuda.empty_cache()

    real_u8 = to_uint8(real_images).cpu()
    fake_u8 = to_uint8(fake_pixels)  # already on CPU from decode_latents

    # 2. metrics
    fid_score, cmmd = compute_metrics(real_u8, fake_u8, device)

    # 4. log scores + a few samples to wandb, and keep a grid on disk
    SAMPLE_DIR.mkdir(exist_ok=True)
    grid_path = SAMPLE_DIR / f"epoch_{epoch:03d}.png"
    save_image(fake_pixels[:16] * 0.5 + 0.5, grid_path, nrow=4)

    wandb.log({
        "eval/fid": fid_score,
        "eval/cmmd": cmmd,
        "eval/gen_time": gen_time,                      # total seconds for the batch
        "eval/gen_time_per_image": gen_time_per_image,  # seconds per image
        "eval/samples": wandb.Image(str(grid_path)),
        "epoch": epoch,
    })
    print(f"Epoch {epoch} | FID: {fid_score:.3f} | CMMD: {cmmd:.3f} "
          f"| gen: {gen_time:.1f}s ({gen_time_per_image*1000:.0f} ms/img)")
    return fid_score


def train():
    device = "cuda" if t.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    wandb.init(
        project="saidl-diffusion",
        name="baseline-dit-b8",  # change this when testing RACD
        entity="bhuvaneshreddy-bits-pilani",
        config={
            "architecture": "DiT-B/8",
            "dataset": "landscapes",
            "epochs": 100,
            "mode": "online",
        },
    )

    EPOCHS = 100             # total epochs to train up to
    BATCH_SIZE = 16          # latents are tiny, so we can fit a bigger batch
    ACCUMULATION_STEPS = 2   # effective batch size stays at 32
    RESUME = True            # continue from the last checkpoint if one exists

    # 1. Latents (built once, then reused) instead of re-encoding every epoch
    cache = build_latent_cache(device)
    latents = cache["latents"]                  # [N, 4, 32, 32], fp16, on CPU
    eval_real = cache["real_pixels"].to(device) # reference set for the metrics

    train_dataset = t.utils.data.TensorDataset(latents)
    dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                            num_workers=2, pin_memory=True)

    # 2. Init Models (VAE kept around for decoding during evaluation)
    vae = FrozenVAE("stabilityai/sd-vae-ft-ema", device)
    # decode one sample at a time -> keeps VAE peak memory low on the 6GB card
    vae.autoencoder.enable_slicing()

    # DiT-B/8 config
    model = DiT(hidden_size=768, patch_size=8, num_heads=12, num_blocks=12, num_classes=1, in_channels=4).to(device)
    optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scaler = GradScaler()  # mixed-precision gradient scaler

    # 3. Schedule prep
    alpha_bar = get_noise_schedule(num_timesteps=1000, device=device)

    CHECKPOINT_DIR.mkdir(exist_ok=True)
    best_fid = float("inf")
    start_epoch = 0

    # optional resume. dit_resume.pt holds full state (model+optim+scaler+epoch)
    # for seamless continuation; if only the older weights-only dit_latest.pt
    # exists we load the weights and restart the optimizer (small warmup cost).
    resume_path = CHECKPOINT_DIR / "dit_resume.pt"
    if RESUME and resume_path.exists():
        ckpt = t.load(resume_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scaler.load_state_dict(ckpt["scaler"])
        best_fid = ckpt.get("best_fid", best_fid)
        start_epoch = ckpt["epoch"] + 1
        print(f"Resumed full state from epoch {ckpt['epoch']} -> starting at {start_epoch}")
    elif RESUME and (CHECKPOINT_DIR / "dit_latest.pt").exists():
        model.load_state_dict(t.load(CHECKPOINT_DIR / "dit_latest.pt", map_location=device))
        start_epoch = 50  # 49 epochs (0..49) already done; continue from here
        print(f"Resumed weights only (fresh optimizer) -> starting at epoch {start_epoch}")

    # 4. The Loop
    model.train()
    for epoch in range(start_epoch, EPOCHS):
        epoch_loss = 0
        optimizer.zero_grad()
        for step, (latents_batch,) in enumerate(dataloader):
            # cached latents are fp16 on CPU -> move to GPU and back to fp32
            latents_batch = latents_batch.to(device, non_blocking=True).float()
            B = latents_batch.shape[0]

            # roll random timesteps
            ts = t.randint(0, 1000, (B,), device=device, dtype=t.long)

            # unconditional dummy labels
            y = t.zeros(B, dtype=t.long, device=device)

            # sample noise and corrupt
            noise = t.randn_like(latents_batch)
            noisy_latents = q_sample(latents_batch, ts, noise, alpha_bar)

            # predict + loss under mixed precision
            with autocast("cuda"):
                predicted_noise = model(noisy_latents, ts, y)
                loss = F.mse_loss(predicted_noise, noise)

            # log the un-scaled loss so the curve is comparable across runs
            wandb.log({"train/loss": loss.item(), "epoch": epoch})

            scaler.scale(loss / ACCUMULATION_STEPS).backward()
            if (step + 1) % ACCUMULATION_STEPS == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            epoch_loss += loss.item()

            if step % 10 == 0:
                print(f"Epoch {epoch} | Step {step} | Loss: {loss.item():.4f}")

        avg_loss = epoch_loss / len(dataloader)
        wandb.log({"train/epoch_loss": avg_loss, "epoch": epoch})
        print(f"Epoch {epoch} finished | Avg Loss: {avg_loss:.4f}")

        # weights-only checkpoint for inference (generate.py / train2.py)
        t.save(model.state_dict(), CHECKPOINT_DIR / "dit_latest.pt")
        # full state so we can resume cleanly next time
        t.save({"epoch": epoch, "model": model.state_dict(),
                "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(),
                "best_fid": best_fid}, CHECKPOINT_DIR / "dit_resume.pt")

        # periodic FID / CLIP evaluation
        if (epoch + 1) % EVAL_EVERY == 0 or (epoch + 1) == EPOCHS:
            fid_score = evaluate(model, vae, alpha_bar, eval_real, device, epoch)
            if fid_score < best_fid:
                best_fid = fid_score
                t.save(model.state_dict(), CHECKPOINT_DIR / "dit_best.pt")

    wandb.finish()
    print(f"Training done. Best FID: {best_fid:.3f}")


if __name__ == "__main__":
    train()
