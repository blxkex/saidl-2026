import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import json
import time
import argparse

import torch as t
from pathlib import Path
from diffusers import DDIMScheduler

from modules import *
from generate import load_models, decode, sample, INFER_STEPS, NUM_TRAIN_TIMESTEPS, LATENT_CACHE
from train import compute_metrics, to_uint8

ROOT = Path(__file__).resolve().parent

# row label -> (refine flag, tau). tau ignored unless refine=="racd"
def build_configs(taus):
    cfgs = [("Baseline DiT", "none", None),
            ("Global Cyclic", "global", None)]
    for tau in taus:
        cfgs.append((f"RACD (tau={tau})", "racd", tau))
    return cfgs


@t.no_grad()
def run_config(label, refine, tau, dit, vae, predictor, scheduler, real_u8, n, device):
    if device == "cuda":
        t.cuda.synchronize()
    start = time.time()
    latents = sample(dit, vae, scheduler, n, device, refine=refine,
                     predictor=predictor, tau=tau if tau is not None else 0.5)
    if device == "cuda":
        t.cuda.synchronize()
    gen_time = time.time() - start

    fake_u8 = to_uint8(decode(vae, latents))
    fid, cmmd = compute_metrics(real_u8, fake_u8, device)
    t.cuda.empty_cache()

    return {"config": label, "fid": round(fid, 3), "cmmd": round(cmmd, 3),
            "gen_time_s": round(gen_time, 2), "ms_per_img": round(gen_time / n * 1000, 1),
            "n": n, "tau": tau}


def run(n=200, taus=(0.3, 0.5, 0.7), seed=0):
    """Run every config, return a list of result dicts."""
    device = "cuda" if t.cuda.is_available() else "cpu"
    dit, vae, predictor = load_models(device)
    scheduler = DDIMScheduler(num_train_timesteps=NUM_TRAIN_TIMESTEPS)
    real_u8 = to_uint8(t.load(LATENT_CACHE)["real_pixels"])[:n].cpu()

    results = []
    for label, refine, tau in build_configs(taus):
        if refine == "racd" and predictor is None:
            results.append({"config": label, "fid": None, "cmmd": None,
                            "gen_time_s": None, "n": n, "tau": tau,
                            "note": "no difficulty_predictor.pt"})
            continue
        t.manual_seed(seed)
        results.append(run_config(label, refine, tau, dit, vae, predictor,
                                  scheduler, real_u8, n, device))
    return results


def latex_rows(results):
    lines = []
    for r in results:
        if r["fid"] is None:
            lines.append(f"{r['config']} & - & - & - \\\\")
        else:
            lines.append(f"{r['config']} & {r['fid']:.2f} & {r['cmmd']:.2f} "
                         f"& {r['gen_time_s']:.2f} \\\\")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description="Benchmark all sampling configs.")
    ap.add_argument("--n", type=int, default=200, help="samples per config (<=200)")
    ap.add_argument("--taus", type=float, nargs="+", default=[0.3, 0.5, 0.7])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", type=str, default=None, help="optional path to dump json")
    args = ap.parse_args()

    print(f"Running {args.n} samples/config ...")
    results = run(n=args.n, taus=tuple(args.taus), seed=args.seed)

    # human table
    print(f"\n{'Config':18s} | {'FID':>8s} | {'CMMD':>6s} | {'Time(s)':>7s} | {'ms/img':>6s}")
    print("-" * 60)
    for r in results:
        if r["fid"] is None:
            print(f"{r['config']:18s} | {'-':>8s} | {'-':>6s} | {'-':>7s} | {'-':>6s}")
        else:
            print(f"{r['config']:18s} | {r['fid']:8.3f} | {r['cmmd']:6.3f} | "
                  f"{r['gen_time_s']:7.2f} | {r['ms_per_img']:6.1f}")

    print("\n--- LaTeX rows ---")
    print(latex_rows(results))

    print("\n--- JSON ---")
    print(json.dumps(results, indent=2))

    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=2))
        print(f"\nSaved {args.json}")

    return results


if __name__ == "__main__":
    main()
