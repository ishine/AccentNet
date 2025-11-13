#!/usr/bin/env python3
"""
Cache 80-bin log-mel spectrograms for utterances listed in the manifest.

Reads audio from the 16 kHz workspace (data/audio16) if available, falling
back to the original paths. Each spectrogram is saved as a Torch tensor to
`data/features/mels/{split}/{utt_id}.pt`.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator

import numpy as np
import torch
import torchaudio
from scipy.io import wavfile


@dataclass(frozen=True)
class ManifestRow:
    utt_id: str
    split: str
    accent: str
    path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pre-compute log-mel features.")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/manifest.csv"),
        help="Manifest CSV produced by prepare_manifest.py.",
    )
    parser.add_argument(
        "--audio-root",
        type=Path,
        default=Path("data/audio16"),
        help="Directory with 16 kHz WAV files named `<utt_id>.wav`.",
    )
    parser.add_argument(
        "--fallback-root",
        type=Path,
        default=Path("."),
        help="Repository root; used when audio-root does not contain a file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/features/mels"),
        help="Destination directory for cached torchaudio tensors.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N items (for smoke tests).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate features even if the output already exists.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch waves before mel computation (1 = per-file).",
    )
    return parser.parse_args()


def read_manifest(manifest_path: Path) -> Iterator[ManifestRow]:
    with manifest_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            yield ManifestRow(
                utt_id=row["utt_id"],
                split=row["split"],
                accent=row["accent"],
                path=Path(row["path"]),
            )


def load_waveform(
    row: ManifestRow,
    audio_root: Path,
    fallback_root: Path,
) -> torch.Tensor:
    audio_path = audio_root / f"{row.utt_id}.wav"
    if audio_path.exists():
        path = audio_path
    else:
        path = fallback_root / row.path
    sr, data = wavfile.read(path)
    if data.ndim > 1:
        data = data.mean(axis=1)
    if data.dtype != np.float32:
        if data.dtype.kind in {"i", "u"}:
            info = np.iinfo(data.dtype)
            scale = float(max(abs(info.min), abs(info.max)))
            data = data.astype(np.float32) / max(scale, 1.0)
        else:
            data = data.astype(np.float32)
    waveform = torch.from_numpy(data).unsqueeze(0)
    if sr != 16000:
        waveform = torchaudio.functional.resample(waveform, sr, 16000)
    return waveform


def build_mel_transform() -> torchaudio.transforms.MelSpectrogram:
    return torchaudio.transforms.MelSpectrogram(
        sample_rate=16000,
        n_fft=1024,
        win_length=int(0.025 * 16000),
        hop_length=int(0.010 * 16000),
        f_min=20.0,
        f_max=7600.0,
        n_mels=80,
        window_fn=torch.hann_window,
        power=2.0,
        center=True,
        pad_mode="reflect",
        norm="slaney",
        mel_scale="slaney",
    )


def compute_log_mel(mel_transform: torchaudio.transforms.MelSpectrogram, waveform: torch.Tensor) -> torch.Tensor:
    mel = mel_transform(waveform)
    mel = torch.clamp(mel, min=1e-5)
    log_mel = torch.log(mel)
    return log_mel.squeeze(0)


def main() -> None:
    args = parse_args()

    rows = list(read_manifest(args.manifest))
    if args.limit is not None:
        rows = rows[: args.limit]

    mel_transform = build_mel_transform()
    torch.set_grad_enabled(False)

    output_root = args.output_dir
    counts: Dict[str, int] = {}
    processed = 0

    for row in rows:
        out_path = output_root / row.split / f"{row.utt_id}.pt"
        if out_path.exists() and not args.overwrite:
            counts[row.split] = counts.get(row.split, 0) + 1
            continue

        waveform = load_waveform(row, args.audio_root, args.fallback_root)
        mel = compute_log_mel(mel_transform, waveform)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"mel": mel, "accent": row.accent}, out_path)

        counts[row.split] = counts.get(row.split, 0) + 1
        processed += 1
        if processed % 100 == 0:
            print(f"Processed {processed} utterances...")

    print("Mel feature cache complete.")
    for split, total in sorted(counts.items()):
        print(f"{split}: {total}")


if __name__ == "__main__":
    main()
