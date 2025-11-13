#!/usr/bin/env python3
"""
Resample utterances listed in the manifest to a uniform sample rate.

The script reads `data/manifest.csv`, converts any non-16 kHz audio to
16 kHz mono WAV using ffmpeg, and copies already-16 kHz files into a
workspace directory. Output filenames follow the `utt_id` key so they
align with downstream features.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from math import gcd
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly


TARGET_SR = 16000


@dataclass(frozen=True)
class ManifestRow:
    utt_id: str
    path: Path
    orig_sample_rate: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Resample audio to 16 kHz.")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/manifest.csv"),
        help="CSV manifest produced by prepare_manifest.py.",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("."),
        help="Repository root so relative paths in the manifest resolve.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/audio16"),
        help="Directory that will hold resampled WAV files.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Resample even if the output file already exists.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N rows (useful for smoke-tests).",
    )
    return parser.parse_args()


def read_manifest(manifest_path: Path) -> Iterator[ManifestRow]:
    with manifest_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            yield ManifestRow(
                utt_id=row["utt_id"],
                path=Path(row["path"]),
                orig_sample_rate=int(row["orig_sample_rate"]),
            )


def resample_wav(src: Path, dst: Path, target_sr: int, overwrite: bool) -> None:
    if dst.exists() and not overwrite:
        return

    dst.parent.mkdir(parents=True, exist_ok=True)
    orig_sr, audio = wavfile.read(src)

    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    if np.issubdtype(audio.dtype, np.integer):
        max_val = np.iinfo(audio.dtype).max
        audio = audio.astype(np.float32) / max_val
    else:
        audio = audio.astype(np.float32)

    if orig_sr == target_sr:
        data = audio
    else:
        factor = gcd(orig_sr, target_sr)
        up = target_sr // factor
        down = orig_sr // factor
        data = resample_poly(audio, up, down)

    data = np.clip(data, -1.0, 1.0)
    int16 = (data * 32767.0).astype(np.int16)
    wavfile.write(dst, target_sr, int16)


def copy_file(src: Path, dst: Path, overwrite: bool) -> None:
    if dst.exists() and not overwrite:
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    data = src.read_bytes()
    dst.write_bytes(data)


def process_manifest(rows: Iterable[ManifestRow], root: Path, output_dir: Path, overwrite: bool) -> None:
    missing = 0
    converted = 0
    copied = 0

    for row in rows:
        src = root / row.path
        if not src.exists():
            missing += 1
            print(f"[missing] {row.utt_id}: {src}")
            continue
        dst = output_dir / f"{row.utt_id}.wav"
        if row.orig_sample_rate == TARGET_SR:
            copy_file(src, dst, overwrite)
            copied += 1
        else:
            resample_wav(src, dst, TARGET_SR, overwrite)
            converted += 1

    print(f"Converted: {converted}")
    print(f"Copied (already {TARGET_SR}): {copied}")
    if missing:
        print(f"Missing files: {missing}")


def main() -> None:
    args = parse_args()
    rows = list(read_manifest(args.manifest))
    if args.limit is not None:
        rows = rows[: args.limit]
    process_manifest(rows, args.root.resolve(), args.output_dir, args.overwrite)


if __name__ == "__main__":
    main()
