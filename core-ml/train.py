import os
import time

import torch as t
import torch.nn as nn
import torch.nn.functional as F

import hydra
from omegaconf import DictConfig, OmegaConf

from tqdm.auto import tqdm
from torch.utils.tensorboard import SummaryWriter

from positional_embeddings import *
from attention_heads import *
from transformer_blocks import *
from data_preprocess import DataPreprocessor


# attention setup.
def build_attention(cfg: DictConfig, ctx_len: int, dim: int, heads: int) -> nn.Module:
    a = cfg.model.attention
    if a.type == "mha":
        return MaskedMultiHeadedAttention(heads, ctx_len, dim)

    if a.type != "flexible":
        raise ValueError(f"unknown attention.type {a.type!r} (use 'mha' or 'flexible')")

    kw = {}
    if a.variant == "SWA":
        kw["window_size"] = a.window_size
    if a.variant == "GQA":
        kw["groups"] = a.groups
    if a.pe == "RPE":
        kw["max_distance"] = a.max_distance

    return FlexibleAttentionBlock(
        pe=a.pe, variant=a.variant, dim=dim, seq_len=ctx_len, heads=heads, **kw
    )


def build_model(cfg: DictConfig, vocab_size: int) -> ModularTransformer:
    ctx_len = cfg.model.context_len
    dim = cfg.model.embed_dim
    heads = cfg.model.n_heads
    n_layers = cfg.model.n_layers
    mode = cfg.model.mode

    n_attn = n_layers // 2 if mode == "alternating" else n_layers
    attention_blocks = [
        build_attention(cfg, ctx_len, dim, heads) for _ in range(n_attn)
    ]

    conv_cfg = {
        "kernel_size": cfg.model.conv.kernel_size,
        "padding": cfg.model.conv.padding,
    }

    return ModularTransformer(
        ctx_len=ctx_len,
        dim=dim,
        n_layers=n_layers,
        vocab_size=vocab_size,
        attention_blocks=attention_blocks,
        mode=mode,
        conv_cfg=conv_cfg,
        use_abs_pe=cfg.model.use_abs_pe,
    )


@t.no_grad()
def evaluate(model, loader, device, vocab_size):
    """Full pass over a held-out split. Returns (avg_loss, perplexity)."""
    model.eval()
    total_loss = 0.0
    n_batches = 0
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs).view(-1, vocab_size)
        labels = labels.view(-1)
        loss = F.cross_entropy(outputs, labels)
        total_loss += loss.item()
        n_batches += 1
    model.train()
    avg_loss = total_loss / max(n_batches, 1)
    perplexity = t.exp(t.tensor(min(avg_loss, 20.0))).item()
    return avg_loss, perplexity


@hydra.main(version_base=None, config_path="configs", config_name="config")
def train(cfg: DictConfig):
    # Setup Device
    device = t.device(cfg.training.device if t.cuda.is_available() else "cpu")

    # -------------------------------------------------------------
    # Setup Data and Model
    # -------------------------------------------------------------
    preprocessor = DataPreprocessor(
        dataset_path=cfg.data.dataset_path,
        dataset_name=cfg.data.dataset_name,
        context_len=cfg.model.context_len,
        batch_size=cfg.training.batch_size,
    )
    train_loader = preprocessor.get_dataloader("train")
    val_loader = preprocessor.get_dataloader("validation")
    vocab_size = preprocessor.tokenizer.vocab_size

    model = build_model(cfg, vocab_size).to(device)
    optimizer = t.optim.AdamW(model.parameters(), lr=cfg.training.lr)

    n_params = sum(p.numel() for p in model.parameters())

    # -------------------------------------------------------------
    # Logging setup (plain text + TensorBoard)
    # -------------------------------------------------------------
    arch = (
        f"mode={cfg.model.mode} | attn={cfg.model.attention.type}"
        f"({cfg.model.attention.pe}/{cfg.model.attention.variant})"
        if cfg.model.attention.type == "flexible"
        else f"mode={cfg.model.mode} | attn=mha"
    )

    epochs = cfg.training.epochs
    grad_accum_steps = max(1, cfg.training.grad_accum_steps)
    tokens_per_step = cfg.training.batch_size * cfg.model.context_len
    eff_batch = cfg.training.batch_size * grad_accum_steps

    # TensorBoard logs under <original cwd>/runs so `tensorboard --logdir runs`
    # (or %tensorboard --logdir runs in a notebook) finds every run.
    log_dir = os.path.join(hydra.utils.get_original_cwd(), "runs")
    writer = SummaryWriter(log_dir=log_dir)
    writer.add_text("config", OmegaConf.to_yaml(cfg))

    # Weights & Biases (optional, gated by cfg.wandb.enabled). Lazy import so the
    # dep is only required when actually used.
    use_wandb = cfg.wandb.enabled
    if use_wandb:
        import wandb

        wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            name=cfg.wandb.name,
            mode=cfg.wandb.mode,
            config=OmegaConf.to_container(cfg, resolve=True),
        )

    print("=" * 70)
    print(f"Transformer Training — device={device.type.upper()} | {arch}")
    print(
        f"dim={cfg.model.embed_dim} heads={cfg.model.n_heads} "
        f"layers={cfg.model.n_layers} | params={n_params:,}"
    )
    print(
        f"dataset={cfg.data.dataset_name} | context={cfg.model.context_len} | "
        f"batch={cfg.training.batch_size} x{grad_accum_steps} = {eff_batch} eff | "
        f"lr={cfg.training.lr}"
    )
    print(f"TensorBoard log_dir: {log_dir}")
    print(f"W&B: {'enabled (' + cfg.wandb.project + ')' if use_wandb else 'disabled'}")
    print("=" * 70)

    # -------------------------------------------------------------
    # Training Loop
    # -------------------------------------------------------------
    best_loss = float("inf")
    best_val_ppl = float("inf")
    global_step = 0
    n_batches = len(train_loader)

    if device.type == "cuda":
        t.cuda.reset_peak_memory_stats(device)

    epoch_bar = tqdm(range(epochs), desc="Epochs", unit="epoch")
    for epoch in epoch_bar:
        model.train()
        total_loss = 0.0
        epoch_start = time.perf_counter()
        tokens_seen = 0

        optimizer.zero_grad()
        batch_bar = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{epochs}",
            leave=False,
            unit="batch",
        )
        for batch_idx, (inputs, labels) in enumerate(batch_bar):
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)

            outputs = outputs.view(-1, vocab_size)
            labels = labels.view(-1)

            loss = F.cross_entropy(outputs, labels)

            # scale so accumulated grads average over the micro-batches
            (loss / grad_accum_steps).backward()

            # step once per grad_accum_steps micro-batches; flush the tail too
            if (batch_idx + 1) % grad_accum_steps == 0 or batch_idx + 1 == n_batches:
                optimizer.step()
                optimizer.zero_grad()

            total_loss += loss.item()
            tokens_seen += tokens_per_step
            global_step += 1

            elapsed = time.perf_counter() - epoch_start
            tok_s = tokens_seen / elapsed if elapsed > 0 else 0.0
            run_avg = total_loss / (batch_idx + 1)
            batch_bar.set_postfix(loss=f"{loss.item():.4f}", avg=f"{run_avg:.4f}")

            writer.add_scalar("train/loss_step", loss.item(), global_step)
            writer.add_scalar("train/tok_per_s", tok_s, global_step)
            if use_wandb:
                wandb.log(
                    {"train/loss_step": loss.item(), "train/tok_per_s": tok_s},
                    step=global_step,
                )

        avg_loss = total_loss / n_batches
        # Clip perplexity for logging so we don't overflow on poor inits
        perplexity = t.exp(t.tensor(min(avg_loss, 20.0))).item()
        best_loss = min(best_loss, avg_loss)

        epoch_time = time.perf_counter() - epoch_start
        epoch_tok_s = tokens_seen / epoch_time if epoch_time > 0 else 0.0

        # Validation pass (held-out split) -> loss + downstream perplexity.
        val_loss, val_ppl = evaluate(model, val_loader, device, vocab_size)
        best_val_ppl = min(best_val_ppl, val_ppl)

        # Peak GPU memory since training start (resident allocator high-water mark).
        peak_mem_mb = (
            t.cuda.max_memory_allocated(device) / (1024**2)
            if device.type == "cuda"
            else 0.0
        )

        writer.add_scalar("train/loss_epoch", avg_loss, epoch + 1)
        writer.add_scalar("train/perplexity", perplexity, epoch + 1)
        writer.add_scalar("train/epoch_tok_per_s", epoch_tok_s, epoch + 1)
        writer.add_scalar("train/epoch_time_s", epoch_time, epoch + 1)
        writer.add_scalar("val/loss", val_loss, epoch + 1)
        writer.add_scalar("val/perplexity", val_ppl, epoch + 1)
        writer.add_scalar("system/peak_gpu_mem_mb", peak_mem_mb, epoch + 1)
        if use_wandb:
            wandb.log(
                {
                    "train/loss_epoch": avg_loss,
                    "train/perplexity": perplexity,
                    "train/epoch_tok_per_s": epoch_tok_s,
                    "train/epoch_time_s": epoch_time,
                    "val/loss": val_loss,
                    "val/perplexity": val_ppl,
                    "system/peak_gpu_mem_mb": peak_mem_mb,
                    "epoch": epoch + 1,
                },
                step=global_step,
            )

        epoch_bar.set_postfix(loss=f"{avg_loss:.4f}", val_ppl=f"{val_ppl:.2f}")
        tqdm.write(
            f"Epoch {epoch + 1:>3}/{epochs} | loss {avg_loss:.4f} | "
            f"val_loss {val_loss:.4f} | val_ppl {val_ppl:8.2f} | "
            f"{epoch_tok_s:,.0f} tok/s | {epoch_time:.1f}s | "
            f"{peak_mem_mb:.0f}MB | best_val_ppl {best_val_ppl:.2f}"
        )

    # Save checkpoint. Name per run so concurrent/sequential runs don't clobber
    # each other's weights locally.
    run_name = (wandb.run.name if use_wandb and wandb.run.name else cfg.wandb.name) or "run"
    save_dir = os.path.join(hydra.utils.get_original_cwd(), "checkpoints")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{run_name}.pt")

    t.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": OmegaConf.to_container(cfg, resolve=True),
            "vocab_size": vocab_size,
        },
        save_path,
    )

    writer.close()
    if use_wandb:

        final_peak_mem_mb = (
            t.cuda.max_memory_allocated(device) / (1024**2)
            if device.type == "cuda"
            else 0.0
        )
        wandb.summary["n_params"] = n_params
        wandb.summary["best_train_loss"] = best_loss
        wandb.summary["best_val_ppl"] = best_val_ppl
        wandb.summary["peak_gpu_mem_mb"] = final_peak_mem_mb
        wandb.summary["arch"] = arch

        # Log weights as a W&B Artifact -> stored server-side, survives Colab
        # disconnects, and tied to this run (artifact name = run name).
        artifact = wandb.Artifact(
            name=run_name,
            type="model",
            metadata={
                "arch": arch,
                "n_params": n_params,
                "best_val_ppl": best_val_ppl,
            },
        )
        artifact.add_file(save_path)
        wandb.log_artifact(artifact)
        wandb.finish()

    print("=" * 70)
    print(f"Training complete. Best loss: {best_loss:.4f}")
    print(f"Checkpoint: {save_path}")
    print("Generate with: python inference.py --interactive")
    print("=" * 70)


if __name__ == "__main__":
    train()
