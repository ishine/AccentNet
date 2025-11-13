#!/usr/bin/env python3
"""
Generate a unified manifest covering British-accent VCTK speakers and
Indian-accent speakers from L2-ARCTIC plus VCTK.

The manifest includes one row per utterance with fields that downstream
pipelines (feature extraction, batching) can consume.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Tuple
import wave


# Speaker-based splits (keep accents stratified).
BRITISH_TRAIN = {
    "p225",
    "p226",
    "p227",
    "p228",
    "p229",
    "p230",
    "p231",
    "p232",
    "p233",
    "p236",
    "p239",
    "p240",
    "p243",
    "p244",
    "p250",
    "p254",
    "p256",
    "p257",
}
BRITISH_VAL = {
    "p258",
    "p259",
    "p273",
    "p274",
    "p276",
    "p277",
    "p278",
}
BRITISH_TEST = {
    "p267",
    "p268",
    "p269",
    "p270",
    "p279",
    "p282",
    "p286",
    "p287",
}

INDIAN_TRAIN = {
    "ASI",
    "SVBI",
    "p248",
    "accentdb_indian_s01",
    "accentdb_bangla_s01",
    "accentdb_malayalam_s01",
    "accentdb_malayalam_s02",
    "accentdb_telugu_s01",
}
INDIAN_VAL = {
    "RRBI",
    "accentdb_indian_s02",
    "accentdb_bangla_s02",
    "accentdb_malayalam_s03",
}
INDIAN_TEST = {
    "p251",
    "p376",
    "accentdb_telugu_s02",
    "accentdb_odiya_s01",
}

L2_SPEAKERS = {"ASI", "SVBI", "RRBI"}

ACCENTDB_SPEAKER_DIRS: Dict[str, Path] = {
    "accentdb_indian_s01": Path("accentDB_extended/data/indian/speaker_01"),
    "accentdb_indian_s02": Path("accentDB_extended/data/indian/speaker_02"),
    "accentdb_bangla_s01": Path("accentDB_extended/data/bangla/speaker_01"),
    "accentdb_bangla_s02": Path("accentDB_extended/data/bangla/speaker_02"),
    "accentdb_malayalam_s01": Path("accentDB_extended/data/malayalam/speaker_01"),
    "accentdb_malayalam_s02": Path("accentDB_extended/data/malayalam/speaker_02"),
    "accentdb_malayalam_s03": Path("accentDB_extended/data/malayalam/speaker_03"),
    "accentdb_telugu_s01": Path("accentDB_extended/data/telugu/speaker_01"),
    "accentdb_telugu_s02": Path("accentDB_extended/data/telugu/speaker_02"),
    "accentdb_odiya_s01": Path("accentDB_extended/data/odiya/speaker_01"),
}

FIELDNAMES = [
    "utt_id",
    "dataset",
    "speaker_id",
    "accent",
    "split",
    "path",
    "text",
    "orig_sample_rate",
]


@dataclass(frozen=True)
class Utterance:
    utt_id: str
    dataset: str
    speaker_id: str
    accent: str
    split: str
    path: Path
    text: str
    orig_sample_rate: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build accent manifest.")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("."),
        help="Repository root containing vctk/ and ASI/RRBI/SVBI folders.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/manifest.csv"),
        help="Destination CSV manifest.",
    )
    return parser.parse_args()


def read_vctk_speaker_info(speaker_info_path: Path) -> Dict[str, str]:
    accent_by_speaker: Dict[str, str] = {}
    with speaker_info_path.open("r", encoding="utf-8") as f:
        next(f)  # skip header
        for raw_line in f:
            parts = raw_line.strip().split()
            if not parts:
                continue
            speaker_id = f"p{int(parts[0]):03d}"
            accent = parts[3]
            accent_by_speaker[speaker_id] = accent
    return accent_by_speaker


def iter_vctk_utterances(
    vctk_root: Path,
    accent_by_speaker: Dict[str, str],
    speaker_split: Dict[str, Tuple[str, str]],
) -> Iterator[Utterance]:
    wav_root = vctk_root / "wav48"
    txt_root = vctk_root / "txt"
    for speaker_dir in sorted(wav_root.iterdir()):
        speaker_id = speaker_dir.name
        if speaker_id not in speaker_split:
            continue
        accent_label, split = speaker_split[speaker_id]
        sample_rate = 48000
        for wav_path in sorted(speaker_dir.glob("*.wav")):
            rel = wav_path.relative_to(vctk_root.parent.parent.parent)
            txt_path = txt_root / speaker_id / (wav_path.stem + ".txt")
            if not txt_path.exists():
                raise FileNotFoundError(f"Missing transcript for {wav_path}")
            text = txt_path.read_text(encoding="utf-8").strip()
            yield Utterance(
                utt_id=f"vctk_{speaker_id}_{wav_path.stem}",
                dataset="vctk",
                speaker_id=speaker_id,
                accent=accent_label,
                split=split,
                path=rel,
                text=text,
                orig_sample_rate=sample_rate,
            )


def iter_l2_utterances(
    repo_root: Path,
    speaker_id: str,
    split: str,
) -> Iterator[Utterance]:
    speaker_root = repo_root / speaker_id
    wav_root = speaker_root / "wav"
    txt_root = speaker_root / "transcript"
    sample_rate = 16000
    for wav_path in sorted(wav_root.glob("*.wav")):
        txt_path = txt_root / (wav_path.stem + ".txt")
        if not txt_path.exists():
            raise FileNotFoundError(f"Missing transcript for {wav_path}")
        text = txt_path.read_text(encoding="utf-8").strip()
        yield Utterance(
            utt_id=f"l2_{speaker_id}_{wav_path.stem}",
            dataset="l2_arctic",
            speaker_id=speaker_id,
            accent="indian",
            split=split,
            path=wav_path.relative_to(repo_root),
            text=text,
            orig_sample_rate=sample_rate,
        )


def get_wav_sample_rate(path: Path) -> int:
    with wave.open(str(path), "rb") as wav_file:
        return wav_file.getframerate()


def iter_accentdb_utterances(
    repo_root: Path,
    speaker_id: str,
    split: str,
) -> Iterator[Utterance]:
    speaker_dir = repo_root / ACCENTDB_SPEAKER_DIRS[speaker_id]
    for wav_path in sorted(speaker_dir.glob("*.wav")):
        sample_rate = get_wav_sample_rate(wav_path)
        text = ""  # transcripts are not provided in AccentDB Extended
        yield Utterance(
            utt_id=f"accentdb_{speaker_id}_{wav_path.stem}",
            dataset="accentdb",
            speaker_id=speaker_id,
            accent="indian",
            split=split,
            path=wav_path.relative_to(repo_root),
            text=text,
            orig_sample_rate=sample_rate,
        )


def build_manifest(args: argparse.Namespace) -> List[Utterance]:
    repo_root: Path = args.root.resolve()

    vctk_root = repo_root / "vctk" / "VCTK-Corpus" / "VCTK-Corpus"
    speaker_info = read_vctk_speaker_info(vctk_root / "speaker-info.txt")

    speaker_split: Dict[str, Tuple[str, str]] = {}

    for speaker_id in BRITISH_TRAIN:
        speaker_split[speaker_id] = ("british", "train")
    for speaker_id in BRITISH_VAL:
        speaker_split[speaker_id] = ("british", "val")
    for speaker_id in BRITISH_TEST:
        speaker_split[speaker_id] = ("british", "test")

    for speaker_id in INDIAN_TRAIN:
        speaker_split[speaker_id] = ("indian", "train")
    for speaker_id in INDIAN_VAL:
        speaker_split[speaker_id] = ("indian", "val")
    for speaker_id in INDIAN_TEST:
        speaker_split[speaker_id] = ("indian", "test")

    unused_vctk = {
        speaker for speaker, accent in speaker_info.items() if speaker not in speaker_split
    }
    # Validate that our selected VCTK speakers have the expected accents.
    for speaker_id, (label, _) in speaker_split.items():
        if speaker_id.startswith("p"):
            actual = speaker_info.get(speaker_id)
            if actual is None:
                raise ValueError(f"Speaker {speaker_id} missing from speaker-info.txt")
            if label == "british" and actual != "English":
                raise ValueError(f"Expected English accent for {speaker_id}, got {actual}")
            if label == "indian" and actual != "Indian":
                raise ValueError(f"Expected Indian accent for {speaker_id}, got {actual}")

    rows: List[Utterance] = []
    rows.extend(iter_vctk_utterances(vctk_root, speaker_info, speaker_split))

    for speaker_id in sorted(INDIAN_TRAIN | INDIAN_VAL | INDIAN_TEST):
        if speaker_id.startswith("p"):
            # Already covered via VCTK iteration.
            continue
        split = (
            "train"
            if speaker_id in INDIAN_TRAIN
            else "val"
            if speaker_id in INDIAN_VAL
            else "test"
        )
        if speaker_id in L2_SPEAKERS:
            rows.extend(iter_l2_utterances(repo_root, speaker_id, split))
        elif speaker_id in ACCENTDB_SPEAKER_DIRS:
            rows.extend(iter_accentdb_utterances(repo_root, speaker_id, split))
        else:
            raise ValueError(f"Unhandled speaker id {speaker_id}")

    return rows


def write_manifest(rows: Iterable[Utterance], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    counts: Counter[str] = Counter()
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "utt_id": row.utt_id,
                    "dataset": row.dataset,
                    "speaker_id": row.speaker_id,
                    "accent": row.accent,
                    "split": row.split,
                    "path": row.path.as_posix(),
                    "text": row.text,
                    "orig_sample_rate": row.orig_sample_rate,
                }
            )
            counts[f"{row.accent}:{row.split}"] += 1
    for key, value in sorted(counts.items()):
        print(f"{key}: {value}")


def main() -> None:
    args = parse_args()
    rows = build_manifest(args)
    write_manifest(rows, args.output.resolve())


if __name__ == "__main__":
    main()
