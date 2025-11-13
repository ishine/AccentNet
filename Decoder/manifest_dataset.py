"""Dataset utilities for loading embeddings from a JSONL manifest."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from dataset import _pad_or_trim, _quantize_durations


@dataclass(frozen=True)
class ManifestEntry:
    dataset: str
    accent: str
    speaker_id: str
    utterance_id: str
    path: Path


def _load_manifest(manifest_path: Path, root_dir: Path) -> List[ManifestEntry]:
    entries: List[ManifestEntry] = []
    with manifest_path.open("r") as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            rel_path = Path(data["path"])
            entries.append(
                ManifestEntry(
                    dataset=data.get("dataset", ""),
                    accent=data.get("accent", ""),
                    speaker_id=str(data.get("speaker_id", "")),
                    utterance_id=str(data.get("utterance_id", "")),
                    path=(root_dir / rel_path),
                )
            )
    return entries


def _group_by_utterance(
    entries: Iterable[ManifestEntry],
) -> Dict[str, List[ManifestEntry]]:
    grouped: Dict[str, List[ManifestEntry]] = {}
    for entry in entries:
        grouped.setdefault(entry.utterance_id, []).append(entry)
    return grouped


def _gather_pairs(
    source_entries: Sequence[ManifestEntry],
    target_lookup: Dict[str, List[ManifestEntry]],
    allowed_utterances: Optional[Sequence[str]] = None,
) -> List[Tuple[ManifestEntry, ManifestEntry]]:
    allow_set = set(str(u) for u in allowed_utterances) if allowed_utterances else None
    pairs: List[Tuple[ManifestEntry, ManifestEntry]] = []
    for src in source_entries:
        if allow_set is not None and src.utterance_id not in allow_set:
            continue
        target_candidates = target_lookup.get(src.utterance_id, [])
        for tgt in target_candidates:
            pairs.append((src, tgt))
    return pairs


def _load_numpy(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Missing numpy file: {path}")
    return np.load(path)


def _load_optional_numpy(path: Path) -> Optional[np.ndarray]:
    return np.load(path) if path.exists() else None


def _load_accent_embedding(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Missing accent.pt file at {path}")
    blob = torch.load(path, map_location="cpu")
    if isinstance(blob, dict) and "embedding" in blob:
        return blob["embedding"].detach().cpu().numpy()
    if isinstance(blob, torch.Tensor):
        return blob.detach().cpu().numpy()
    raise ValueError(f"Unexpected accent.pt contents at {path}")


class ManifestAccentPairDataset(Dataset):
    """Pair source and target entries from a manifest for accent conversion.

    Each dataset item contains:
        - content_emb: [seq, 768]
        - speaker_emb: [192]
        - accent_emb: [seq or 1?, 256] from target speaker (broadcast to seq)
        - pitch: [seq] (converted to mel-length later)
        - energy: [seq]
        - duration: [seq] (aligned to mel if mel available)
        - mel_target: optional [mel, 80] if found on disk

    Notes:
        The current embeddings_clean layout stores:
            cv.npy     -> content
            spk.npy    -> speaker
            lf0.npy    -> pitch contour (per frame)
            lf0i.npy   -> intensity (proxy for energy)
            accent.pt  -> accent embedding dict
            mel.npy    -> optional mel spectrogram (if created separately)
            duration.npy/dur.npy -> optional durations
    """

    def __init__(
        self,
        manifest_path: str,
        root_dir: str,
        source_accent: str,
        target_accent: str,
        source_dataset: Optional[str] = None,
        target_dataset: Optional[str] = None,
        source_speaker: Optional[str] = None,
        target_speaker: Optional[str] = None,
        allowed_utterances: Optional[Sequence[str]] = None,
        require_mel: bool = False,
    ) -> None:
        self.root_dir = Path(root_dir)
        manifest = _load_manifest(Path(manifest_path), self.root_dir)

        def _filter(
            entries: Sequence[ManifestEntry],
            accent: str,
            dataset: Optional[str],
            speaker: Optional[str],
        ) -> List[ManifestEntry]:
            out: List[ManifestEntry] = []
            for e in entries:
                if e.accent.lower() != accent.lower():
                    continue
                if dataset and e.dataset.lower() != dataset.lower():
                    continue
                if speaker and e.speaker_id.lower() != speaker.lower():
                    continue
                out.append(e)
            return out

        source_entries = _filter(
            manifest, source_accent, source_dataset, source_speaker
        )
        target_entries = _filter(
            manifest, target_accent, target_dataset, target_speaker
        )
        if not source_entries:
            raise ValueError(
                f"No manifest entries found for source accent '{source_accent}'."
            )
        if not target_entries:
            raise ValueError(
                f"No manifest entries found for target accent '{target_accent}'."
            )

        target_lookup = _group_by_utterance(target_entries)
        self.pairs = _gather_pairs(source_entries, target_lookup, allowed_utterances)

        if not self.pairs:
            raise ValueError(
                "No matching source/target pairs found. "
                "Ensure utterance IDs overlap between accents or provide allowed_utterances."
            )

        self.require_mel = require_mel

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        src, tgt = self.pairs[idx]
        src_features = self._load_features(src)
        tgt_features = self._load_features(tgt, load_mel=True)

        content_np = src_features["content"]
        speaker_np = src_features["speaker"]
        pitch_np = src_features["pitch"]
        energy_np = src_features["energy"]
        accent_np = tgt_features["accent"]
        seq_len = content_np.shape[0]
        content_emb = torch.from_numpy(content_np).float()
        speaker_emb = torch.from_numpy(speaker_np).float()
        accent_seq_np = np.tile(accent_np, (seq_len, 1))
        accent_emb = torch.from_numpy(accent_seq_np).float()

        # Align lengths
        content_len = seq_len
        pitch_np = _pad_or_trim(pitch_np, content_len)
        energy_np = _pad_or_trim(energy_np, content_len)

        # Duration handling
        duration = tgt_features.get("duration")
        mel = tgt_features.get("mel")
        if duration is None:
            duration = np.ones(content_len, dtype=np.float32)
        else:
            duration = _pad_or_trim(duration, content_len)

        mel_len = mel.shape[0] if mel is not None else int(duration.sum())
        pitch = torch.from_numpy(_pad_or_trim(pitch_np, mel_len)).float()
        energy = torch.from_numpy(_pad_or_trim(energy_np, mel_len)).float()
        duration = torch.from_numpy(_quantize_durations(duration, mel_len)).float()

        item = {
            "content_emb": content_emb,
            "speaker_emb": speaker_emb,
            "accent_emb": torch.from_numpy(
                _pad_or_trim(accent_seq_np, content_len)
            ).float(),
            "pitch": pitch,
            "energy": energy,
            "duration": duration,
        }

        if mel is not None:
            item["mel_target"] = torch.from_numpy(mel).float()
        elif self.require_mel:
            raise FileNotFoundError(
                f"Mel spectrogram not available for target path {tgt.path}"
            )

        return item

    def _load_features(
        self, entry: ManifestEntry, load_mel: bool = False
    ) -> Dict[str, np.ndarray]:
        base = entry.path
        if not base.exists():
            raise FileNotFoundError(f"Expected directory missing: {base}")

        features: Dict[str, np.ndarray] = {
            "content": _load_numpy(base / "cv.npy"),
            "speaker": _load_numpy(base / "spk.npy"),
            "pitch": _load_numpy(base / "lf0.npy"),
            "energy": _load_numpy(base / "lf0i.npy"),
            "accent": _load_accent_embedding(base / "accent.pt"),
        }

        for name in ("duration.npy", "dur.npy"):
            duration_path = base / name
            if duration_path.exists():
                features["duration"] = np.load(duration_path)
                break

        if load_mel:
            for name in ("mel.npy", "mel_spectrogram.npy", "mel.pt"):
                mel_path = base / name
                if mel_path.exists():
                    if mel_path.suffix == ".pt":
                        tensor = torch.load(mel_path, map_location="cpu")
                        if isinstance(tensor, torch.Tensor):
                            features["mel"] = tensor.detach().cpu().numpy()
                        else:
                            raise ValueError(f"Unsupported mel.pt contents at {mel_path}")
                    else:
                        features["mel"] = np.load(mel_path)
                    break

        return features
