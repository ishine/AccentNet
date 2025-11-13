#!/usr/bin/env python3
"""
CLI entry-point for training the accent encoder.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from accent_encoder.train import TrainConfig, main


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the accent encoder.")
    parser.add_argument("--manifest", type=Path, default=Path("data/manifest.csv"))
    parser.add_argument("--features", type=Path, default=Path("data/features/mels"))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--total-steps", type=int, default=2000)
    parser.add_argument("--warmup-steps", type=int, default=200)
    parser.add_argument("--log-dir", type=Path, default=Path("runs/accent_encoder"))
    parser.add_argument("--ckpt-dir", type=Path, default=Path("checkpoints/accent_encoder"))
    parser.add_argument("--lambda-grl", type=float, default=1.0)
    parser.add_argument("--speaker-loss-weight", type=float, default=0.5)
    parser.add_argument("--validate-every", type=int, default=500)
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument("--device", type=str, default=None, help="Override device (cpu/cuda/mps).")
    return parser.parse_args()


def run() -> None:
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    config = TrainConfig(
        manifest_path=args.manifest,
        feature_root=args.features,
        batch_size=args.batch_size,
        num_workers=args.workers,
        lr=args.lr,
        total_steps=args.total_steps,
        warmup_steps=args.warmup_steps,
        log_dir=args.log_dir,
        checkpoint_dir=args.ckpt_dir,
        lambda_grl=args.lambda_grl,
        speaker_loss_weight=args.speaker_loss_weight,
        validate_every=args.validate_every,
        save_every=args.save_every,
        device=device,
    )
    main(config)


if __name__ == "__main__":
    import torch

    run()
