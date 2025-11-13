"""
Data utilities for the accent encoder.

The dataset reads cached log-mel tensors produced by `scripts/extract_mels.py`
and exposes padded minibatches for the training loop.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import torch
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence


@dataclass(frozen=True)
class ManifestEntry:
    utt_id: str
    speaker_id: str
    accent: str
    split: str
    feature_path: Path


class LabelEncoder:
    """Utility to map string labels (accent, speaker) to integer ids."""

    def __init__(self, labels: Iterable[str]) -> None:
        unique = sorted(set(labels))
        self._label_to_idx: Dict[str, int] = {label: idx for idx, label in enumerate(unique)}
        self._idx_to_label: List[str] = unique

    def encode(self, label: str) -> int:
        try:
            return self._label_to_idx[label]
        except KeyError as exc:
            raise KeyError(f"Unknown label {label!r}") from exc

    def decode(self, index: int) -> str:
        return self._idx_to_label[index]

    @property
    def size(self) -> int:
        return len(self._idx_to_label)

    @property
    def labels(self) -> Sequence[str]:
        return self._idx_to_label


def read_manifest(manifest_path: Path, feature_root: Path) -> Dict[str, List[ManifestEntry]]:
    """Parse manifest.csv and resolve feature paths."""
    entries: Dict[str, List[ManifestEntry]] = {"train": [], "val": [], "test": []}
    with manifest_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            split = row["split"]
            if split not in entries:
                continue
            utt_id = row["utt_id"]
            feature_path = feature_root / split / f"{utt_id}.pt"
            if not feature_path.exists():
                raise FileNotFoundError(
                    f"Expected feature tensor at {feature_path} for utterance {utt_id}"
                )
            entries[split].append(
                ManifestEntry(
                    utt_id=utt_id,
                    speaker_id=row["speaker_id"],
                    accent=row["accent"],
                    split=split,
                    feature_path=feature_path,
                )
            )
    return entries


class AccentDataset(Dataset):
    """Torch dataset returning mel tensors plus labels."""

    def __init__(
        self,
        entries: Sequence[ManifestEntry],
        accent_encoder: LabelEncoder,
        speaker_encoder: LabelEncoder,
    ) -> None:
        self.entries = list(entries)
        self.accent_encoder = accent_encoder
        self.speaker_encoder = speaker_encoder

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        entry = self.entries[idx]
        record = torch.load(entry.feature_path)
        mel = record["mel"]  # shape (n_mels, frames)
        if mel.dim() != 2:
            raise ValueError(f"Expected 2D mel tensor, got shape {mel.shape}")
        mel = mel.transpose(0, 1).contiguous()  # (frames, n_mels)

        accent_id = self.accent_encoder.encode(entry.accent)
        speaker_id = self.speaker_encoder.encode(entry.speaker_id)

        return {
            "mel": mel,
            "accent_id": torch.tensor(accent_id, dtype=torch.long),
            "speaker_id": torch.tensor(speaker_id, dtype=torch.long),
            "utt_id": entry.utt_id,
        }


def collate_batch(batch: Sequence[Mapping[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Pad mel sequences in time dimension and stack labels."""
    mels = [item["mel"] for item in batch]
    lengths = torch.tensor([mel.size(0) for mel in mels], dtype=torch.long)
    padded = pad_sequence(mels, batch_first=True)  # (B, T_max, n_mels)
    accent_ids = torch.stack([item["accent_id"] for item in batch])
    speaker_ids = torch.stack([item["speaker_id"] for item in batch])
    utt_ids = [item["utt_id"] for item in batch]
    return {
        "mel": padded,
        "lengths": lengths,
        "accent_id": accent_ids,
        "speaker_id": speaker_ids,
        "utt_id": utt_ids,
    }


def build_label_encoders(entries: Dict[str, List[ManifestEntry]]) -> Tuple[LabelEncoder, LabelEncoder]:
    accents = [e.accent for split_entries in entries.values() for e in split_entries]
    speakers = [e.speaker_id for split_entries in entries.values() for e in split_entries]
    accent_encoder = LabelEncoder(accents)
    speaker_encoder = LabelEncoder(speakers)
    return accent_encoder, speaker_encoder


def create_dataloaders(
    manifest_path: Path,
    feature_root: Path,
    batch_size: int = 16,
    num_workers: int = 4,
) -> Tuple[Dict[str, DataLoader], LabelEncoder, LabelEncoder]:
    """
    Build torch DataLoaders for train/val/test splits.

    Returns data loaders along with the label encoders so callers can look up
    vocabulary sizes for the classifier heads.
    """
    entries = read_manifest(manifest_path, feature_root)
    accent_encoder, speaker_encoder = build_label_encoders(entries)

    datasets = {
        split: AccentDataset(split_entries, accent_encoder, speaker_encoder)
        for split, split_entries in entries.items()
    }

    loaders = {
        split: DataLoader(
            dataset,
            batch_size=batch_size if split == "train" else batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            pin_memory=False,
            collate_fn=collate_batch,
        )
        for split, dataset in datasets.items()
    }
    return loaders, accent_encoder, speaker_encoder
