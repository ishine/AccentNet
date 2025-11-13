#!/usr/bin/env python3
"""
Compute accent centroids from cached embeddings.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute accent centroids from embeddings.")
    parser.add_argument("--manifest", type=Path, default=Path("data/manifest.csv"))
    parser.add_argument(
        "--embedding-dir", type=Path, default=Path("data/features/embeddings"), help="Root directory created by extract_embeddings.py"
    )
    parser.add_argument("--splits", nargs="+", default=["val"], help="Splits to include (e.g., val test).")
    parser.add_argument("--output", type=Path, default=Path("data/features/accent_centroids.pt"))
    parser.add_argument("--json", type=Path, default=Path("data/features/accent_centroids.json"))
    return parser.parse_args()


def collect_utterances(manifest_path: Path, splits: List[str]) -> Dict[str, List[Tuple[str, str]]]:
    accent_to_utts: Dict[str, List[Tuple[str, str]]] = {}
    with manifest_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            split = row["split"]
            if split not in splits:
                continue
            accent = row["accent"]
            accent_to_utts.setdefault(accent, []).append((row["utt_id"], split))
    return accent_to_utts


def main() -> None:
    args = parse_args()
    accent_utts = collect_utterances(args.manifest, args.splits)
    centroids: Dict[str, torch.Tensor] = {}

    for accent, entries in accent_utts.items():
        vectors = []
        missing = 0
        for utt_id, split in entries:
            path = args.embedding_dir / split / f"{utt_id}.pt"
            if not path.exists():
                missing += 1
                continue
            data = torch.load(path, map_location="cpu")
            vec = torch.nn.functional.normalize(data["embedding"], dim=0)
            vectors.append(vec)
        if not vectors:
            raise ValueError(f"No embeddings found for accent {accent} in splits {args.splits}")
        stacked = torch.stack(vectors, dim=0)
        centroid = torch.nn.functional.normalize(stacked.mean(dim=0), dim=0)
        centroids[accent] = centroid
        print(f"{accent}: {len(vectors)} embeddings (missing {missing})")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(centroids, args.output)
    args.json.parent.mkdir(parents=True, exist_ok=True)
    with args.json.open("w", encoding="utf-8") as f:
        json.dump({accent: tensor.tolist() for accent, tensor in centroids.items()}, f)
    print(f"Saved centroids to {args.output} and {args.json}")


if __name__ == "__main__":
    main()
