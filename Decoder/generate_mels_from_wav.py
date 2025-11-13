#!/usr/bin/env python3
# mel_to_wav.py
# Reconstruct a waveform from natural-log mel spectrogram using Griffin–Lim.
# Uses the metadata JSON produced by wav_to_mel.py to ensure consistent params.

import argparse, json, numpy as np, soundfile as sf, librosa

def mel_to_wav(logmel, meta, n_iter=80):
    assert meta["layout"] == "n_mels_T"
    n_mels, T = logmel.shape

    # Rebuild mel filter and pseudo-inverse
    mel_basis = librosa.filters.mel(
        sr=meta["sr"], n_fft=meta["n_fft"], n_mels=meta["n_mels"],
        fmin=meta["fmin"], fmax=meta["fmax"],
        htk=meta["mel_htk"], norm=("slaney" if meta["mel_norm"] else None)
    )
    inv_mel = np.linalg.pinv(mel_basis)

    # Undo log, project to linear spectrogram magnitude^power
    mel_power = np.exp(logmel)  # undo ln
    lin_power = np.maximum(inv_mel @ mel_power, 1e-10)

    # If we encoded |S|**power, recover magnitude
    mag = lin_power ** (1.0 / meta["power"])

    # Griffin–Lim
    y = librosa.griffinlim(
        S=mag,
        n_iter=n_iter,
        hop_length=meta["hop_length"],
        win_length=meta["win_length"],
        window="hann",
        center=True,
        momentum=0.0,
        init="random",
        random_state=0,
    )
    # Normalize to -1..1
    y = y / (np.max(np.abs(y)) + 1e-9)
    return y.astype(np.float32)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mel", required=True, help="Input log-mel .npy (shape [n_mels, T] or [T, n_mels])")
    ap.add_argument("--meta", required=True, help="Metadata .json from wav_to_mel.py")
    ap.add_argument("--out", required=True, help="Output .wav path")
    ap.add_argument("--gl_iters", type=int, default=80, help="Griffin–Lim iterations (60–120 reasonable)")
    args = ap.parse_args()

    logmel = np.load(args.mel)
    with open(args.meta) as f:
        meta = json.load(f)

    # Accept [T, n_mels] too
    if logmel.shape[0] != meta["n_mels"] and logmel.shape[1] == meta["n_mels"]:
        logmel = logmel.T

    y = mel_to_wav(logmel.astype(np.float32), meta, n_iter=args.gl_iters)
    sf.write(args.out, y, meta["sr"])
    print(f"Wrote {args.out}  (sr={meta['sr']}, dur≈{len(y)/meta['sr']:.2f}s)")
