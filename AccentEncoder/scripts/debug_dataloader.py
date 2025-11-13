#!/usr/bin/env python3
"""
Quick sanity-check for AccentDataset/DataLoader.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import sys


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from accent_encoder.data import create_dataloaders


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect one minibatch.")
    parser.add_argument("--manifest", type=Path, default=Path("data/manifest.csv"))
    parser.add_argument("--features", type=Path, default=Path("data/features/mels"))
    parser.add_argument("--batch-size", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    loaders, accent_encoder, speaker_encoder = create_dataloaders(
        manifest_path=args.manifest,
        feature_root=args.features,
        batch_size=args.batch_size,
        num_workers=0,
    )
    batch = next(iter(loaders["train"]))
    print("mel:", batch["mel"].shape)
    print("lengths:", batch["lengths"])
    print("accent ids:", batch["accent_id"])
    print("speaker ids:", batch["speaker_id"])
    print("utt ids:", batch["utt_id"])
    print("accent labels:", [accent_encoder.decode(i.item()) for i in batch["accent_id"]])
    print("speakers:", [speaker_encoder.decode(i.item()) for i in batch["speaker_id"]])


if __name__ == "__main__":
    main()
