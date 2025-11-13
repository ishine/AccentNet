"""
Training loop for the accent encoder.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.optim as optim

try:
    from torch.utils.tensorboard import SummaryWriter
except ModuleNotFoundError:
    SummaryWriter = None  # type: ignore[assignment]

from .data import create_dataloaders
from .model import AccentEncoder, GradientReversal


@dataclass
class TrainConfig:
    manifest_path: Path = Path("data/manifest.csv")
    feature_root: Path = Path("data/features/mels")
    batch_size: int = 16
    num_workers: int = 4
    lr: float = 3e-4
    warmup_steps: int = 1000
    total_steps: int = 20000
    grad_clip: float = 1.0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    log_dir: Path = Path("runs/accent_encoder")
    checkpoint_dir: Path = Path("checkpoints/accent_encoder")
    accent_loss_weight: float = 1.0
    speaker_loss_weight: float = 0.5
    lambda_grl: float = 1.0
    validate_every: int = 1000
    save_every: int = 1000


def build_model(accent_classes: int, speaker_classes: int, config: TrainConfig) -> AccentEncoder:
    model = AccentEncoder(accent_classes=accent_classes, speaker_classes=speaker_classes)
    return model.to(config.device)


def get_scheduler(optimizer: optim.Optimizer, config: TrainConfig):
    def lr_lambda(step: int) -> float:
        if step < config.warmup_steps:
            return (step + 1) / max(config.warmup_steps, 1)
        progress = (step - config.warmup_steps) / max(config.total_steps - config.warmup_steps, 1)
        return max(0.1, 0.5 * (1 + math.cos(math.pi * progress)))

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def train_epoch(
    model: AccentEncoder,
    loaders: Dict[str, torch.utils.data.DataLoader],
    accent_grl: GradientReversal,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    scheduler,
    config: TrainConfig,
    writer,
    start_step: int = 0,
) -> int:
    model.train()
    step = start_step
    for batch in loaders["train"]:
        mel = batch["mel"].to(config.device)
        lengths = batch["lengths"].to(config.device)
        accent_id = batch["accent_id"].to(config.device)
        speaker_id = batch["speaker_id"].to(config.device)

        embedding, accent_logits, speaker_logits = model(mel, lengths)
        accent_loss = criterion(accent_logits, accent_id)
        speaker_logits = model.speaker_classifier(accent_grl(embedding))
        speaker_loss = criterion(speaker_logits, speaker_id)

        loss = config.accent_loss_weight * accent_loss + config.speaker_loss_weight * speaker_loss
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()
        scheduler.step()

        if writer is not None and step % 100 == 0:
            lr = scheduler.get_last_lr()[0]
            writer.add_scalar("train/loss", loss.item(), step)
            writer.add_scalar("train/accent_loss", accent_loss.item(), step)
            writer.add_scalar("train/speaker_loss", speaker_loss.item(), step)
            writer.add_scalar("train/lr", lr, step)

        if config.validate_every and step % config.validate_every == 0 and step > 0:
            validate(model, loaders["val"], criterion, config, writer, step)

        if config.save_every and step % config.save_every == 0 and step > 0:
            save_checkpoint(model, optimizer, step, config)

        step += 1
        if step >= config.total_steps:
            break
    return step


@torch.no_grad()
def validate(
    model: AccentEncoder,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    config: TrainConfig,
    writer,
    step: int,
) -> None:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    for batch in loader:
        mel = batch["mel"].to(config.device)
        lengths = batch["lengths"].to(config.device)
        labels = batch["accent_id"].to(config.device)
        _, logits, _ = model(mel, lengths)
        loss = criterion(logits, labels)
        total_loss += loss.item() * mel.size(0)
        preds = logits.argmax(dim=-1)
        total_correct += (preds == labels).sum().item()
        total_examples += mel.size(0)
    avg_loss = total_loss / max(total_examples, 1)
    acc = total_correct / max(total_examples, 1)
    if writer is not None:
        writer.add_scalar("val/loss", avg_loss, step)
        writer.add_scalar("val/accent_acc", acc, step)
    print(f"[val] step={step} loss={avg_loss:.4f} acc={acc:.4f}")


def save_checkpoint(model: AccentEncoder, optimizer: optim.Optimizer, step: int, config: TrainConfig) -> None:
    config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    path = config.checkpoint_dir / f"step_{step}.pt"
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "step": step}, path)
    print(f"Saved checkpoint to {path}")


def main(config: TrainConfig) -> None:
    loaders, accent_encoder, speaker_encoder = create_dataloaders(
        config.manifest_path,
        config.feature_root,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
    )
    model = build_model(accent_encoder.size, speaker_encoder.size, config)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config.lr)
    scheduler = get_scheduler(optimizer, config)
    if SummaryWriter is None:
        print("TensorBoard not available; proceeding without logging.")
        writer = None
    else:
        writer = SummaryWriter(log_dir=config.log_dir)
    accent_grl = GradientReversal(lambda_=config.lambda_grl)

    step = 0
    while step < config.total_steps:
        step = train_epoch(
            model,
            loaders,
            accent_grl,
            criterion,
            optimizer,
            scheduler,
            config,
            writer,
            start_step=step,
        )
        if config.save_every and step % config.save_every == 0:
            save_checkpoint(model, optimizer, step, config)


if __name__ == "__main__":
    main(TrainConfig())
