#!/usr/bin/env python3
"""Create train/val/test utterance split files from the JSONL manifest."""

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Set, Tuple


def load_manifest(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with path.open("r") as f:
        for line in f:
            if not line.strip():
                continue
            rows.append(json.loads(line))
    return rows


def collect_ids(
    rows: List[Dict],
    accent: str,
    dataset: str = "",
    speaker: str = "",
) -> Set[str]:
    accent = accent.lower()
    dataset = dataset.lower()
    speaker = speaker.lower()
    result: Set[str] = set()
    for row in rows:
        if row.get("accent", "").lower() != accent:
            continue
        if dataset and row.get("dataset", "").lower() != dataset:
            continue
        if speaker and str(row.get("speaker_id", "")).lower() != speaker:
            continue
        #result.add(str(row.get("utterance_id")))
        result.add(f"{row.get('dataset')}|{row.get('speaker_id')}|{row.get('utterance_id')}")
    return result


def split_ids(ids: List[str], train: float, val: float, seed: int) -> Tuple[List[str], List[str], List[str]]:
    ids_sorted = sorted(ids)
    random.Random(seed).shuffle(ids_sorted)
    n_total = len(ids_sorted)
    n_train = int(n_total * train)
    n_val = int(n_total * val)
    train_ids = ids_sorted[:n_train]
    val_ids = ids_sorted[n_train:n_train + n_val]
    test_ids = ids_sorted[n_train + n_val:]
    return train_ids, val_ids, test_ids


def write_ids(path: Path, ids: List[str]) -> None:
    if not ids:
        return
    with path.open("w") as f:
        for utt in ids:
            f.write(f"{utt}\n")


def main():
    parser = argparse.ArgumentParser(description="Create split id files from manifest.")
    parser.add_argument("--manifest", required=True, help="Path to manifest.jsonl")
    parser.add_argument("--output_dir", default="Decoder/metadata_manifest", help="Directory for output split files")
    parser.add_argument("--source_accent", required=True, help="Source accent name")
    parser.add_argument("--target_accent", required=True, help="Target accent name")
    parser.add_argument("--source_dataset", default="", help="Optional source dataset filter")
    parser.add_argument("--target_dataset", default="", help="Optional target dataset filter")
    parser.add_argument("--source_speaker", default="", help="Optional source speaker filter")
    parser.add_argument("--target_speaker", default="", help="Optional target speaker filter")
    parser.add_argument("--train_ratio", type=float, default=0.7, help="Train split ratio")
    parser.add_argument("--val_ratio", type=float, default=0.15, help="Validation split ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    rows = load_manifest(Path(args.manifest))
    src_ids = collect_ids(rows, args.source_accent, args.source_dataset, args.source_speaker)
    tgt_ids = collect_ids(rows, args.target_accent, args.target_dataset, args.target_speaker)
    overlap = sorted(src_ids & tgt_ids)

    if not overlap:
        raise ValueError("No overlapping utterance IDs found for the given filters.")

    train_ids, val_ids, test_ids = split_ids(overlap, args.train_ratio, args.val_ratio, args.seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    write_ids(out_dir / "train.txt", train_ids)
    write_ids(out_dir / "val.txt", val_ids)
    write_ids(out_dir / "test.txt", test_ids)

    print(f"Total overlapping IDs: {len(overlap)}")
    print(f"Train: {len(train_ids)}")
    print(f"Val:   {len(val_ids)}")
    print(f"Test:  {len(test_ids)}")
    print(f"Files written to {out_dir}")


if __name__ == "__main__":
    main()
