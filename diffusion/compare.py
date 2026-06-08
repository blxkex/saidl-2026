import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import torch as t
from pathlib import Path
from diffusers import DDIMScheduler
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from modules import *
from generate import load_models, decode, sample, NUM_TRAIN_TIMESTEPS

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "refine_eval"

# CLI mode -> generate.sample() refine flag
MODES = {"baseline": "none", "global": "global", "racd": "racd"}


def main():
    ap = argparse.ArgumentParser(description="Side-by-side comparison grid across modes.")
    ap.add_argument("--modes", nargs="+", choices=MODES, default=["baseline", "global"],
                    help="modes to compare (2 or 3): baseline global racd")
    ap.add_argument("--n", type=int, default=4, help="images per mode")
    ap.add_argument("--seed", type=int, default=None,
                    help="base rng seed (random if omitted; same for all modes)")
    ap.add_argument("--tau", type=float, default=0.5, help="racd difficulty threshold")
    ap.add_argument("--out", type=str, default=None, help="output png path")
    args = ap.parse_args()

    seed = args.seed if args.seed is not None else t.randint(0, 2**31 - 1, (1,)).item()

    device = "cuda" if t.cuda.is_available() else "cpu"
    print(f"Using device: {device} | seed {seed}")
    OUT_DIR.mkdir(exist_ok=True)

    dit, vae, predictor = load_models(device)
    if "racd" in args.modes and predictor is None:
        raise SystemExit("racd needs checkpoints/difficulty_predictor.pt (train train2.py)")

    scheduler = DDIMScheduler(num_train_timesteps=NUM_TRAIN_TIMESTEPS)

    # one row per mode; same seed each row -> matching base latent per column
    rows = []
    for mode in args.modes:
        t.manual_seed(seed)
        latents = sample(dit, vae, scheduler, args.n, device,
                         refine=MODES[mode], predictor=predictor, tau=args.tau)
        imgs = (decode(vae, latents).clamp(-1, 1) + 1) / 2  # [-1,1] -> [0,1]
        rows.append(imgs.permute(0, 2, 3, 1).numpy())       # [n,H,W,C]
        print(f"generated {mode}")

    nrows, ncols = len(args.modes), args.n
    fig, axes = plt.subplots(nrows, ncols, figsize=(2 * ncols, 2 * nrows),
                             squeeze=False)
    for r, mode in enumerate(args.modes):
        for c in range(ncols):
            ax = axes[r][c]
            ax.imshow(rows[r][c])
            ax.set_xticks([]); ax.set_yticks([])
            if c == 0:
                ax.set_ylabel(mode, fontsize=11)
    fig.suptitle(f"seed {seed} | {args.n}/mode", fontsize=10)
    fig.tight_layout()

    name = "_".join(args.modes)
    out = Path(args.out) if args.out else OUT_DIR / f"compare_{name}_seed{seed}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
