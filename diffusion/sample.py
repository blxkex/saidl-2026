import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import torch as t
from pathlib import Path
from diffusers import DDIMScheduler
from torchvision.utils import save_image

from modules import *
from generate import load_models, decode, sample, NUM_TRAIN_TIMESTEPS

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "paper" / "generations"

# label -> generate.sample() refine flag. racd uses --tau.
MODES = [("baseline", "none"), ("global", "global"), ("racd", "racd")]


def gen_one(dit, vae, predictor, scheduler, mode_flag, seed, tau, device):
    # same seed -> same base latent across modes, so the 3 are comparable
    t.manual_seed(seed)
    latents = sample(dit, vae, scheduler, 1, device,
                     refine=mode_flag, predictor=predictor, tau=tau)
    return (decode(vae, latents).clamp(-1, 1) + 1) / 2  # [-1,1] -> [0,1]


def main():
    ap = argparse.ArgumentParser(
        description="Generate baseline/global/racd images into paper/generations/.")
    ap.add_argument("--count", type=int, default=5, help="images per mode")
    ap.add_argument("--tau", type=float, default=0.3, help="racd difficulty threshold")
    ap.add_argument("--seeds", type=int, nargs="+", default=None,
                    help="explicit seeds (default: random, one per image)")
    args = ap.parse_args()

    device = "cuda" if t.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    dit, vae, predictor = load_models(device)
    if predictor is None:
        raise SystemExit("racd needs checkpoints/difficulty_predictor.pt (train train2.py)")

    scheduler = DDIMScheduler(num_train_timesteps=NUM_TRAIN_TIMESTEPS)

    seeds = args.seeds or t.randint(0, 2**31 - 1, (args.count,)).tolist()
    print(f"seeds: {seeds}")

    for seed in seeds:
        for label, flag in MODES:
            img = gen_one(dit, vae, predictor, scheduler, flag, seed, args.tau, device)
            out = OUT_DIR / f"{label}_seed{seed}.png"
            save_image(img, out)
            print(f"Saved {out}")

    print(f"\nDone. {len(seeds) * len(MODES)} images in {OUT_DIR}")


if __name__ == "__main__":
    main()
