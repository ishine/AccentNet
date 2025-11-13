#!/usr/bin/env python3
# wav_to_mel.py
# Convert a WAV file to a natural-log mel spectrogram and save metadata for perfect inversion.

import argparse, json, numpy as np
import librosa

try:
    import soundfile as sf
except ModuleNotFoundError:
    sf = None

def wav_to_logmel(
    wav_path,
    sr=24000,
    n_fft=1024,
    hop_length=256,
    win_length=1024,
    n_mels=80,
    fmin=0,
    fmax=12000,
    mel_norm=True,   # librosa "slaney" norm
    mel_htk=False,   # False = Slaney
    power=1.0,       # 1.0 => magnitude, 2.0 => power; keep 1.0 for stable inversion
    eps=1e-5
):
    # Load mono (prefer soundfile, fallback to librosa)
    if sf is not None:
        y, in_sr = sf.read(wav_path)
        if y.ndim > 1:
            y = y.mean(axis=1)
    else:
        y, in_sr = librosa.load(wav_path, sr=None, mono=True)
    if in_sr != sr:
        y = librosa.resample(y, orig_sr=in_sr, target_sr=sr, res_type="kaiser_best")

    # STFT magnitude or power
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window="hann")) ** power

    # Mel filterbank and projection
    mel_basis = librosa.filters.mel(
        sr=sr, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax,
        htk=mel_htk, norm=("slaney" if mel_norm else None)
    )
    mel = np.dot(mel_basis, S)  # [n_mels, T]

    # Natural-log mel (stable)
    logmel = np.log(mel + eps).astype(np.float32)  # [n_mels, T]

    meta = {
        "sr": sr,
        "n_fft": n_fft,
        "hop_length": hop_length,
        "win_length": win_length,
        "n_mels": n_mels,
        "fmin": float(fmin),
        "fmax": float(fmax),
        "mel_norm": bool(mel_norm),
        "mel_htk": bool(mel_htk),
        "power": float(power),
        "eps": float(eps),
        "layout": "n_mels_T"  # we store [n_mels, T]
    }
    return logmel, meta

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", required=True, help="Input .wav path")
    ap.add_argument("--mel_out", required=True, help="Output .npy for log-mel")
    ap.add_argument("--meta_out", required=True, help="Output .json metadata")
    ap.add_argument("--sr", type=int, default=24000)
    ap.add_argument("--n_fft", type=int, default=1024)
    ap.add_argument("--hop_length", type=int, default=256)
    ap.add_argument("--win_length", type=int, default=1024)
    ap.add_argument("--n_mels", type=int, default=80)
    ap.add_argument("--fmin", type=float, default=0.0)
    ap.add_argument("--fmax", type=float, default=12000.0)
    ap.add_argument("--mel_norm", type=lambda s: s.lower()=="true", default=True)
    ap.add_argument("--mel_htk", type=lambda s: s.lower()=="true", default=False)
    ap.add_argument("--power", type=float, default=1.0)
    ap.add_argument("--eps", type=float, default=1e-5)
    args = ap.parse_args()

    logmel, meta = wav_to_logmel(
        args.wav, sr=args.sr, n_fft=args.n_fft, hop_length=args.hop_length, win_length=args.win_length,
        n_mels=args.n_mels, fmin=args.fmin, fmax=args.fmax, mel_norm=args.mel_norm, mel_htk=args.mel_htk,
        power=args.power, eps=args.eps
    )
    np.save(args.mel_out, logmel)
    with open(args.meta_out, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved mel: {args.mel_out}  meta: {args.meta_out}  shape={logmel.shape}")
