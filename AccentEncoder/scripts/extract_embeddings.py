#!/usr/bin/env python3
"""
Generate accent encoder embeddings for cached mel features.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable

import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from accent_encoder.data import create_dataloaders  # noqa: E402
from accent_encoder.model import AccentEncoder  # noqa: E402
from accent_encoder.train import TrainConfig  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract accent embeddings from a trained checkpoint.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to model checkpoint (.pt).")
    parser.add_argument("--splits", nargs="+", default=["val", "test"], help="Dataset splits to process.")
    parser.add_argument("--manifest", type=Path, default=Path("data/manifest.csv"))
    parser.add_argument("--features", type=Path, default=Path("data/features/mels"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/features/embeddings"))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", type=str, default=None, help="cpu/cuda/mps (auto-detect if omitted).")
    parser.add_argument("--limit", type=int, default=None, help="Optional max utterances per split.")
    return parser.parse_args()


def ensure_device(requested: str | None) -> torch.device:
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main() -> None:
    args = parse_args()
    device = ensure_device(args.device)

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
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()

    output_root = args.output_dir
    output_root.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for split in args.splits:
            if split not in loaders:
                raise ValueError(f"Unknown split '{split}'. Available splits: {list(loaders.keys())}")
            loader = loaders[split]
            split_dir = output_root / split
            split_dir.mkdir(parents=True, exist_ok=True)
            seen = 0
            for batch in loader:
                mel = batch["mel"].to(device)
                lengths = batch["lengths"].to(device)
                utt_ids = batch["utt_id"]

                embeddings, accent_logits, _ = model(mel, lengths)
                embeddings = embeddings.cpu()
                accent_logits = accent_logits.cpu()

                for idx, utt_id in enumerate(utt_ids):
                    out_path = split_dir / f"{utt_id}.pt"
                    torch.save(
                        {
                            "embedding": embeddings[idx],
                            "accent_logits": accent_logits[idx],
                            "accent_id": batch["accent_id"][idx].item(),
                            "speaker_id": batch["speaker_id"][idx].item(),
                        },
                        out_path,
                    )
                    seen += 1
                    if args.limit is not None and seen >= args.limit:
                        break
                if args.limit is not None and seen >= args.limit:
                    break
            print(f"[{split}] saved {seen} embeddings to {split_dir}")


if __name__ == "__main__":
    main()
