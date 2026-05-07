import os

import torch as t
import torch.nn as nn
import torch.nn.functional as F

import hydra
from omegaconf import DictConfig, OmegaConf

from rich.live import Live
from rich.layout import Layout
from rich.panel import Panel
from rich.progress import (
    Progress, 
    SpinnerColumn, 
    BarColumn, 
    TextColumn, 
    TimeElapsedColumn, 
    TimeRemainingColumn
)
from rich.table import Table
from rich.align import Align
from rich.text import Text
from rich import box

from positional_embeddings import *
from attention_heads import *
from transformer_blocks import *
from data_preprocess import DataPreprocessor


def make_layout() -> Layout:
    layout = Layout(name="root")
    
    layout.split(
        Layout(name="header", size=3),
        Layout(name="main", ratio=1),
    )
    layout["main"].split_row(
        Layout(name="left_pane", ratio=2),
        Layout(name="right_pane", ratio=1)
    )
    layout["left_pane"].split_column(
        Layout(name="progress", ratio=1),
        Layout(name="status", ratio=1)
    )
    return layout


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
    vocab_size = preprocessor.tokenizer.vocab_size

    model = ModularTransformer(
        ctx_len=cfg.model.context_len,
        dim=cfg.model.embed_dim,
        heads=cfg.model.n_heads,
        n_layers=cfg.model.n_layers,
        vocab_size=vocab_size,
    ).to(device)

    optimizer = t.optim.AdamW(model.parameters(), lr=cfg.training.lr)
    
    # -------------------------------------------------------------
    # Setup Dashboard (Rich TUI)
    # -------------------------------------------------------------
    layout = make_layout()
    
    header_text = Text(f"🚀 Transformer Training Dashboard - Device: {device.type.upper().strip()}", style="bold white on blue", justify="center")
    layout["header"].update(Panel(header_text, style="black"))
    
    progress = Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(bar_width=None),
        "[progress.percentage]{task.percentage:>3.0f}%",
        "•",
        TimeElapsedColumn(),
        "•",
        TimeRemainingColumn(),
        expand=True
    )
    
    epochs = cfg.training.epochs
    epoch_task = progress.add_task("[bold green]Total Epochs", total=epochs)
    batch_task = progress.add_task("[bold cyan]Current Batch", total=len(train_loader))
    
    layout["progress"].update(Panel(progress, title="[bold white]Progress[/]", border_style="green", padding=(1, 2)))
    
    metrics_table = Table(
        title="Training History", 
        box=box.SIMPLE_HEAVY, 
        expand=True,
        header_style="bold magenta"
    )
    metrics_table.add_column("Epoch", justify="center")
    metrics_table.add_column("Avg Loss", justify="center")
    metrics_table.add_column("Perplexity", justify="center")
    
    layout["right_pane"].update(Panel(metrics_table, title="[bold white]Metrics[/]", border_style="magenta"))
    
    setup_info = f"Dataset: {cfg.data.dataset_name}\nContext Len: {cfg.model.context_len}  |  Batch Size: {cfg.training.batch_size}\nLR: {cfg.training.lr}\nParameters: {sum(p.numel() for p in model.parameters()):,}"
    status_text = Text(f"Initializing training for {epochs} epochs...\n\n{setup_info}", style="italic white")
    layout["status"].update(Panel(Align.center(status_text, vertical="middle"), title="[bold white]System Status[/]", border_style="cyan"))

    # -------------------------------------------------------------
    # Training Loop
    # -------------------------------------------------------------
    with Live(layout, refresh_per_second=10) as live:
        for epoch in range(epochs):
            model.train()
            total_loss = 0.0
            
            progress.reset(batch_task, total=len(train_loader), description=f"[bold cyan]Epoch {epoch+1}/{epochs} Batches")
            status_text = Text(f"Epoch {epoch+1} started. Training on {len(train_loader)} batches...", style="yellow")
            layout["status"].update(Panel(Align.center(status_text, vertical="middle"), title="[bold white]System Status[/]", border_style="cyan"))
            
            for batch_idx, (inputs, labels) in enumerate(train_loader):
                inputs, labels = inputs.to(device), labels.to(device)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                
                # Reshape for loss calculation
                # ModularTransformer outputs raw logits. We use CrossEntropyLoss.
                outputs = outputs.view(-1, vocab_size)
                labels = labels.view(-1)
                
                loss = F.cross_entropy(outputs, labels)
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                
                # Update progress
                progress.advance(batch_task)
                
                # Update live status frequently
                if batch_idx % max(1, len(train_loader) // 20) == 0:
                    current_loss = loss.item()
                    status_msg = f"Epoch: [bold]{epoch+1}/{epochs}[/]\nBatch: [bold]{batch_idx}/{len(train_loader)}[/]\nCurrent Loss: [bold]{current_loss:.4f}[/]\n\nProcessing smoothly..."
                    status_text = Text.from_markup(status_msg, style="cyan")
                    layout["status"].update(Panel(Align.center(status_text, vertical="middle"), title="[bold white]System Status[/]", border_style="cyan"))
            
            avg_loss = total_loss / len(train_loader)
            # Clip perplexity for logging so we don't overflow on poor randomized initializations
            perplexity = t.exp(t.tensor(min(avg_loss, 20.0))).item()
            
            # Log to metrics table
            metrics_table.add_row(f"{epoch+1}/{epochs}", f"{avg_loss:.4f}", f"{perplexity:.2f}")
            
            progress.advance(epoch_task)
            
        # Save checkpoint
        save_dir = os.path.join(hydra.utils.get_original_cwd(), "checkpoints")
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, "model.pt")
        
        t.save({
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": OmegaConf.to_container(cfg, resolve=True),
            "vocab_size": vocab_size,
        }, save_path)
        
        status_text = Text.from_markup(f"[bold green]✨ Training Complete! ✨[/]\nCheckpoint saved to [bold]{save_path}[/]", style="bold green")
        layout["status"].update(Panel(Align.center(status_text, vertical="middle"), title="[bold white]System Status[/]", border_style="cyan"))

if __name__ == "__main__":
    train()
