#!/usr/bin/env python3
"""Inference script for FastSpeech2Accent decoder."""

import argparse
import yaml
import torch
import numpy as np
import pickle
from pathlib import Path

try:
    from .model import FastSpeech2Accent
    from .dataset import _pad_or_trim, _quantize_durations
    from .manifest_dataset import ManifestAccentPairDataset
except ImportError:
    from model import FastSpeech2Accent
    from dataset import _pad_or_trim, _quantize_durations
    from manifest_dataset import ManifestAccentPairDataset


class InferenceConfig:
    """Simple config loader matching TrainConfig interface."""

    def __init__(self, cfg_dict):
        for key, value in cfg_dict.items():
            setattr(self, key, value)


def load_config(path):
    with open(path, "r") as f:
        return InferenceConfig(yaml.safe_load(f))


def load_sample(emb_dir, source_speaker, target_speaker, sentence_id):
    """Load embeddings/prosody for a given source/target pair."""
    emb_dir = Path(emb_dir)
    source_path = emb_dir / source_speaker
    target_path = emb_dir / target_speaker

    content = np.load(source_path / f"{sentence_id}_content.npy")
    speaker = np.load(source_path / f"{sentence_id}_speaker.npy")

    with open(source_path / f"{sentence_id}_prosody.pkl", "rb") as f:
        prosody = pickle.load(f)

    accent = np.load(target_path / f"{sentence_id}_accent.npy")
    mel_target_path = target_path / f"{sentence_id}_mel.npy"
    mel_target = np.load(mel_target_path) if mel_target_path.exists() else None

    seq_len = content.shape[0]
    accent = _pad_or_trim(accent, seq_len)
    duration = _pad_or_trim(prosody["duration"], seq_len)

    mel_len = (
        int(mel_target.shape[0])
        if mel_target is not None
        else int(prosody["pitch"].shape[0])
    )
    pitch = _pad_or_trim(prosody["pitch"], mel_len)
    energy = _pad_or_trim(prosody["energy"], mel_len)
    duration = _quantize_durations(duration, mel_len)

    sample = {
        "content_emb": torch.from_numpy(content).unsqueeze(0).float(),
        "speaker_emb": torch.from_numpy(speaker).unsqueeze(0).float(),
        "accent_emb": torch.from_numpy(accent).unsqueeze(0).float(),
        "pitch": torch.from_numpy(pitch).unsqueeze(0).float(),
        "energy": torch.from_numpy(energy).unsqueeze(0).float(),
        "duration": torch.from_numpy(duration).unsqueeze(0).float(),
    }
    if mel_target is not None:
        sample["mel_target"] = torch.from_numpy(mel_target).unsqueeze(0).float()
    return sample


def main():
    parser = argparse.ArgumentParser(description="Run accent conversion inference.")
    parser.add_argument("--config", required=True, help="Model config YAML path.")
    parser.add_argument("--checkpoint", required=True, help="Trained model checkpoint (.pt).")
    parser.add_argument("--embeddings_dir", required=True, help="Directory with embeddings.")
    parser.add_argument("--source_speaker", required=True, help="Source speaker id.")
    parser.add_argument("--target_speaker", required=True, help="Target speaker id.")
    parser.add_argument("--sentence_id", required=True, help="Sentence identifier.")
    parser.add_argument("--manifest", default=None, help="Optional manifest JSONL to resolve paths automatically.")
    parser.add_argument("--source_accent", default=None, help="Source accent name for manifest lookup.")
    parser.add_argument("--target_accent", default=None, help="Target accent name for manifest lookup.")
    parser.add_argument("--source_dataset", default=None, help="Source dataset filter for manifest lookup.")
    parser.add_argument("--target_dataset", default=None, help="Target dataset filter for manifest lookup.")
    parser.add_argument("--output_mel", default="generated_mel.npy", help="Output path for generated mel spectrogram.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    config = load_config(args.config)
    device = torch.device(args.device)

    model = FastSpeech2Accent(config).to(device)
    state_dict = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    if args.manifest:
        if not args.source_accent or not args.target_accent:
            raise ValueError("Manifest mode requires --source_accent and --target_accent.")
        dataset = ManifestAccentPairDataset(
            manifest_path=args.manifest,
            root_dir=args.embeddings_dir,
            source_accent=args.source_accent or "",
            target_accent=args.target_accent or "",
            source_dataset=args.source_dataset,
            target_dataset=args.target_dataset,
            source_speaker=args.source_speaker,
            target_speaker=args.target_speaker,
            allowed_utterances=[args.sentence_id],
            require_mel=False,
        )
        if len(dataset) == 0:
            raise ValueError("No pairs found for the requested combination.")
        sample = dataset[0]
    else:
        sample = load_sample(
            args.embeddings_dir,
            args.source_speaker,
            args.target_speaker,
            args.sentence_id,
        )

    with torch.no_grad():
        mel_pred, mel_refined = model(
            sample["content_emb"].to(device),
            sample["speaker_emb"].to(device),
            sample["accent_emb"].to(device),
            pitch_target=sample["pitch"].to(device),
            energy_target=sample["energy"].to(device),
            duration_target=sample["duration"].to(device),
        )

    mel_out = mel_refined.squeeze(0).cpu().numpy()
    output_path = Path(args.output_mel)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, mel_out)

    print(f"Generated mel saved to {output_path}")
    print(f"Shape: {mel_out.shape}")

    if "mel_target" in sample:
        target = sample["mel_target"]
        pred_tensor = torch.from_numpy(mel_out).unsqueeze(0).to(target.dtype)
        min_len = min(pred_tensor.size(1), target.size(1))
        if min_len > 0:
            diff = torch.nn.functional.l1_loss(
                pred_tensor[:, :min_len], target[:, :min_len], reduction="mean"
            ).item()
            print(
                f"L1 distance to target mel (first {min_len} frames compared): {diff:.4f}"
            )
        else:
            print("Skipping L1 comparison; no overlapping frames.")


if __name__ == "__main__":
    main()
