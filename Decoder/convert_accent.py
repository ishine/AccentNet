#!/usr/bin/env python3
"""
End-to-end accent conversion inference.

Given a trained FastSpeech2Accent checkpoint and precomputed embeddings,
this script generates a converted mel spectrogram and synthesises audio
with a HiFi-GAN vocoder.

Example:
    python Decoder/convert_accent.py \\
        --config Decoder/config/model.yaml \\
        --checkpoint Decoder/checkpoints/model_epoch_10.pt \\
        --manifest Decoder/embeddings_clean/manifest.jsonl \\
        --embeddings_root Decoder/embeddings_clean \\
        --source_accent indian --source_dataset accentdb --source_speaker speaker_02 \\
        --target_accent english --target_dataset vtck --target_speaker p256 \\
        --utterance_id 135 \\
        --output_dir Decoder/outputs/inference
"""

from __future__ import annotations

import argparse
import yaml
import torch
import numpy as np
from pathlib import Path

import torchaudio
from speechbrain.inference.vocoders import HIFIGAN

torch.set_grad_enabled(False)

try:
    from .model import FastSpeech2Accent
    from .manifest_dataset import ManifestAccentPairDataset
except ImportError:
    from model import FastSpeech2Accent
    from manifest_dataset import ManifestAccentPairDataset


class ConfigWrapper:
    """Container matching TrainConfig interface used during training."""

    def __init__(self, cfg_dict):
        for key, value in cfg_dict.items():
            setattr(self, key, value)


def load_config(path: str) -> ConfigWrapper:
    with open(path, "r") as f:
        return ConfigWrapper(yaml.safe_load(f))


def fetch_sample(args) -> dict:
    dataset = ManifestAccentPairDataset(
        manifest_path=args.manifest,
        root_dir=args.embeddings_root,
        source_accent=args.source_accent,
        target_accent=args.target_accent,
        source_dataset=args.source_dataset,
        target_dataset=args.target_dataset,
        source_speaker=args.source_speaker,
        target_speaker=args.target_speaker,
        allowed_utterances=[args.utterance_id],
        require_mel=False,
    )
    if len(dataset) == 0:
        raise ValueError(
            "No matching source/target pair found. "
            "Check accent/dataset/speaker filters and utterance_id."
        )
    sample = dataset[0]
    # Add batch dimension expected by the model
    for key in ["content_emb", "speaker_emb", "accent_emb", "pitch", "energy", "duration"]:
        sample[key] = sample[key].unsqueeze(0)
    if "mel_target" in sample:
        sample["mel_target"] = sample["mel_target"].unsqueeze(0)
    return sample


def synthesise_audio(mel: np.ndarray, vocoder_id: str, sample_rate: int, device: torch.device) -> torch.Tensor:
    """Run HiFi-GAN to turn mel spectrogram into waveform."""
    vocoder = HIFIGAN.from_hparams(
        source=vocoder_id,
        savedir="pretrained_models/tts-hifigan-ljspeech",
        run_opts={"device": device},
    )
    mel_tensor = torch.from_numpy(mel).unsqueeze(0)  # [1, T, 80]
    mel_tensor = mel_tensor.transpose(1, 2).to(device)  # -> [1, 80, T]
    waveform = vocoder.decode_batch(mel_tensor).cpu().squeeze(0)  # [T]
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)  # [1, T]
    return waveform


def main():
    parser = argparse.ArgumentParser(description="Accent conversion inference with vocoder output.")
    parser.add_argument("--config", required=True, help="Model config YAML path.")
    parser.add_argument("--checkpoint", required=True, help="Trained model checkpoint (.pt).")
    parser.add_argument("--manifest", required=True, help="JSONL manifest describing embeddings.")
    parser.add_argument("--embeddings_root", required=True, help="Root directory containing embeddings (cv.npy, etc.).")
    parser.add_argument("--source_accent", required=True, help="Source accent name in manifest (e.g., indian).")
    parser.add_argument("--target_accent", required=True, help="Target accent name in manifest (e.g., english).")
    parser.add_argument("--source_dataset", default=None, help="Optional source dataset filter (e.g., accentdb).")
    parser.add_argument("--target_dataset", default=None, help="Optional target dataset filter (e.g., vtck).")
    parser.add_argument("--source_speaker", default=None, help="Source speaker identifier.")
    parser.add_argument("--target_speaker", default=None, help="Target speaker identifier.")
    parser.add_argument("--utterance_id", required=True, help="Utterance ID shared by source and target.")
    parser.add_argument("--output_dir", default="Decoder/outputs/inference", help="Directory to save mel and wav.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Device for model/vocoder.")
    parser.add_argument("--vocoder", default="speechbrain/tts-hifigan-ljspeech", help="SpeechBrain HiFi-GAN identifier.")
    parser.add_argument("--sample_rate", type=int, default=22050, help="Sample rate expected by the chosen vocoder.")
    parser.add_argument("--save_mel", action="store_true", help="Save generated mel spectrogram (.npy).")
    args = parser.parse_args()

    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = load_config(args.config)
    model = FastSpeech2Accent(config).to(device)
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state)
    model.eval()

    sample = fetch_sample(args)

    with torch.no_grad():
        mel_pred, mel_refined = model(
            sample["content_emb"].to(device),
            sample["speaker_emb"].to(device),
            sample["accent_emb"].to(device),
            pitch_target=sample["pitch"].to(device),
            energy_target=sample["energy"].to(device),
            duration_target=sample["duration"].to(device),
        )

    mel_np = mel_refined.squeeze(0).cpu().numpy()  # [frames, 80]
    if args.save_mel:
        mel_path = output_dir / f"{args.source_speaker or 'src'}_{args.target_speaker or 'tgt'}_{args.utterance_id}_mel.npy"
        np.save(mel_path, mel_np)
        print(f"Saved mel spectrogram to {mel_path}")

    wav = synthesise_audio(mel_np, args.vocoder, args.sample_rate, device)
    wav_path = output_dir / f"{args.source_speaker or 'src'}_{args.target_speaker or 'tgt'}_{args.utterance_id}.wav"
    torchaudio.save(str(wav_path), wav, args.sample_rate)
    print(f"Saved converted waveform to {wav_path}")

    if "mel_target" in sample:
        target = sample["mel_target"].squeeze(0)
        min_len = min(target.size(0), mel_np.shape[0])
        l1 = torch.nn.functional.l1_loss(
            torch.from_numpy(mel_np[:min_len]), target[:min_len]
        ).item()
        print(f"L1 distance to target mel (first {min_len} frames): {l1:.4f}")


if __name__ == "__main__":
    main()
