#!/usr/bin/env python3
"""
Evaluate speaker leakage by training a simple classifier on frozen embeddings.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn
import torch.optim as optim

from accent_encoder.data import create_dataloaders
from accent_encoder.model import AccentEncoder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Speaker probe for accent encoder.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=Path("data/manifest.csv"))
    parser.add_argument("--features", type=Path, default=Path("data/features/mels"))
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    loaders, accent_le, speaker_le = create_dataloaders(
        manifest_path=args.manifest,
        feature_root=args.features,
        batch_size=args.batch_size,
        num_workers=args.workers,
    )
    model = AccentEncoder(
        accent_classes=accent_le.size,
        speaker_classes=speaker_le.size,
    )
    state = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(state["model"])
    model.to(device)
    model.eval()

    emb_dim = model.embedding[-1].out_features
    probe = nn.Linear(emb_dim, speaker_le.size).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(probe.parameters(), lr=1e-3)

    def get_embeddings(batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        mel = batch["mel"].to(device)
        lengths = batch["lengths"].to(device)
        with torch.no_grad():
            emb, _, _ = model(mel, lengths)
        return emb

    for epoch in range(args.epochs):
        probe.train()
        total_loss = 0.0
        total = 0
        for batch in loaders["train"]:
            emb = get_embeddings(batch)
            labels = batch["speaker_id"].to(device)
            logits = probe(emb)
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * emb.size(0)
            total += emb.size(0)
        print(f"Epoch {epoch+1} train loss: {total_loss/total:.4f}")

    @torch.no_grad()
    def eval_split(split: str) -> None:
        probe.eval()
        correct = 0
        total = 0
        for batch in loaders[split]:
            emb = get_embeddings(batch)
            labels = batch["speaker_id"].to(device)
            logits = probe(emb)
            preds = logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += emb.size(0)
        acc = correct / max(total, 1)
        chance = 1.0 / speaker_le.size
        print(f"{split}: accuracy={acc:.4f} (chance={chance:.4f})")

    eval_split("train")
    eval_split("val")
    eval_split("test")


if __name__ == "__main__":
    main()
