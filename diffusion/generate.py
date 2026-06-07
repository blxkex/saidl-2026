import os
# keep CUDA fragmentation down on the 6GB card (set before torch loads)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import time
import argparse

import torch as t
import torch.nn.functional as F
from pathlib import Path
from diffusers import DDIMScheduler, AutoencoderKL
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from modules import *
from DiT import DiT
from train import compute_metrics, to_uint8   # reuse the FID/CMMD path

ROOT = Path(__file__).resolve().parent
DIT_CKPT = ROOT / "checkpoints" / "dit_best.pt"
PRED_CKPT = ROOT / "checkpoints" / "difficulty_predictor.pt"
LATENT_CACHE = ROOT / "latent_cache.pt"
OUT_DIR = ROOT / "refine_eval"

DIT_KWARGS = dict(hidden_size=768, patch_size=8, num_heads=12,
                  num_blocks=12, num_classes=1, in_channels=4)
VAE_NAME = "stabilityai/sd-vae-ft-ema"
VAE_SCALE = 0.18215

NUM_TRAIN_TIMESTEPS = 1000
INFER_STEPS = 50
REFINE_T = 200      # cyclic refinement re-injects noise up to this timestep
GRID = 4            # 4x4 = 16 patches (latent 32 / patch 8)


def load_models(device):
    dit = DiT(**DIT_KWARGS).to(device)
    dit.load_state_dict(t.load(DIT_CKPT, map_location=device))
    dit.eval()

    vae = AutoencoderKL.from_pretrained(VAE_NAME).to(device).eval()
    vae.enable_slicing()  # decode one image at a time -> low VRAM

    predictor = None
    if PRED_CKPT.exists():
        predictor = DifficultyPredictor(hidden_size=DIT_KWARGS["hidden_size"]).to(device)
        predictor.load_state_dict(t.load(PRED_CKPT, map_location=device))
        predictor.eval()
    return dit, vae, predictor


@t.no_grad()
def racd_mask(dit, predictor, x0, scheduler, tau, device):
    # Ask the difficulty predictor which patches are hard, at the same noise
    # level (REFINE_T) where refinement happens, then make a binary patch mask
    # upsampled to latent resolution. 1 = refine this region, 0 = leave it.
    n = x0.shape[0]
    y = t.zeros(n, dtype=t.long, device=device)
    noise = t.randn_like(x0)
    jump = t.full((n,), REFINE_T, device=device, dtype=t.long)
    x_t = scheduler.add_noise(x0, noise, jump)

    _, features = dit.forward_with_features(x_t, jump, y)   # [n, 16, hidden]
    scores = predictor(features).squeeze(-1)                # [n, 16] in [0,1]
    mask_patch = (scores > tau).float().view(n, 1, GRID, GRID)
    return F.interpolate(mask_patch, scale_factor=32 // GRID, mode="nearest")  # [n,1,32,32]


@t.no_grad()
def sample(dit, vae, scheduler, n, device, refine="none", predictor=None, tau=0.5):
    # refine: "none" (baseline) | "global" (refine everywhere) | "racd" (masked)
    scheduler.set_timesteps(INFER_STEPS, device=device)
    y = t.zeros(n, dtype=t.long, device=device)

    # 1. base generation: pure noise -> clean latent x0
    latents = t.randn(n, 4, 32, 32, device=device)
    for ts in scheduler.timesteps:
        eps = dit(latents, ts.repeat(n), y)
        latents = scheduler.step(eps, ts, latents).prev_sample
    x0 = latents

    # 2. optional cyclic refinement
    if refine != "none":
        if refine == "global":
            mask = t.ones(n, 1, 32, 32, device=device)
        else:
            mask = racd_mask(dit, predictor, x0, scheduler, tau, device)

        # re-noise up to REFINE_T (only masked regions), denoise back, keeping
        # the untouched regions equal to the original clean latent
        noise = t.randn_like(x0)
        jump = t.full((n,), REFINE_T, device=device, dtype=t.long)
        x = scheduler.add_noise(x0, noise, jump)
        x = x * mask + x0 * (1 - mask)
        for ts in scheduler.timesteps:
            if ts > REFINE_T:
                continue
            eps = dit(x, ts.repeat(n), y)
            x = scheduler.step(eps, ts, x).prev_sample
            x = x * mask + x0 * (1 - mask)
        x0 = x

    return x0


@t.no_grad()
def decode(vae, latents):
    imgs = []
    for s in range(0, latents.shape[0], 8):
        z = latents[s:s + 8] / VAE_SCALE
        imgs.append(vae.decode(z).sample.float().cpu())
    return t.cat(imgs, dim=0)


def evaluate_config(name, dit, vae, scheduler, real_u8, device, n,
                    refine="none", predictor=None, tau=0.5):
    if device == "cuda":
        t.cuda.synchronize()
    start = time.time()
    latents = sample(dit, vae, scheduler, n, device, refine, predictor, tau)
    if device == "cuda":
        t.cuda.synchronize()
    gen_time = time.time() - start

    fake_u8 = to_uint8(decode(vae, latents))
    fid, cmmd = compute_metrics(real_u8, fake_u8, device)
    t.cuda.empty_cache()

    print(f"{name:16s} | FID {fid:7.3f} | CMMD {cmmd:7.3f} | "
          f"time {gen_time:6.1f}s ({gen_time / n * 1000:.0f} ms/img)")
    return {"name": name, "fid": fid, "cmmd": cmmd, "time": gen_time, "tau": tau}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=128, help="samples per config")
    ap.add_argument("--taus", type=float, nargs="+", default=[0.3, 0.5, 0.7])
    args = ap.parse_args()

    device = "cuda" if t.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    OUT_DIR.mkdir(exist_ok=True)

    dit, vae, predictor = load_models(device)
    scheduler = DDIMScheduler(num_train_timesteps=NUM_TRAIN_TIMESTEPS)

    # real reference set (same images the baseline used for its metrics)
    real_u8 = to_uint8(t.load(LATENT_CACHE)["real_pixels"])[:args.n].cpu()

    results = []
    results.append(evaluate_config("baseline", dit, vae, scheduler, real_u8, device, args.n, refine="none"))
    results.append(evaluate_config("global_cyclic", dit, vae, scheduler, real_u8, device, args.n, refine="global"))
    if predictor is None:
        print("No difficulty_predictor.pt found -> skipping RACD. Train train2.py first.")
    else:
        for tau in args.taus:
            results.append(evaluate_config(f"racd_t{tau}", dit, vae, scheduler, real_u8,
                                           device, args.n, refine="racd", predictor=predictor, tau=tau))

    # Fidelity (CMMD) vs Compute Time plot
    plt.figure(figsize=(7, 5))
    for r in results:
        plt.scatter(r["time"], r["cmmd"], s=60)
        plt.annotate(r["name"], (r["time"], r["cmmd"]),
                     textcoords="offset points", xytext=(6, 4), fontsize=8)
    plt.xlabel("Compute time (s)")
    plt.ylabel("CMMD (lower = better fidelity)")
    plt.title("Fidelity vs Compute: baseline / global cyclic / RACD")
    plt.grid(True, alpha=0.3)
    plot_path = OUT_DIR / "cmmd_vs_compute.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved plot to {plot_path}")

    # optimal tau among RACD runs = best fidelity (lowest CMMD)
    racd = [r for r in results if r["name"].startswith("racd")]
    if racd:
        best = min(racd, key=lambda r: r["cmmd"])
        print(f"Best RACD by CMMD: tau={best['tau']} (CMMD {best['cmmd']:.3f}, "
              f"time {best['time']:.1f}s)")


if __name__ == "__main__":
    main()
