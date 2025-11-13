import torch
import torch.nn as nn
import numpy as np
import librosa
import pyworld as pw


class ProsodyEncoder(nn.Module):
    """
    Learns to encode F0, energy, and rhythm into both:
      1. frame-level embeddings (B, T, D)
      2. global utterance-level embedding (B, D)
    """

    def __init__(self, n_mels=80, hidden_dim=256, out_dim=128):
        super().__init__()
        in_dim = n_mels + 2  # mel + F0 + energy

        self.conv = nn.Sequential(
            nn.Conv1d(in_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        self.rnn = nn.GRU(
            hidden_dim, hidden_dim // 2,
            num_layers=1, bidirectional=True, batch_first=True
        )

        self.linear = nn.Linear(hidden_dim, out_dim)
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        """
        x: (B, T, in_dim)
        returns:
            frame_emb: (B, T, out_dim)
            global_emb: (B, out_dim)
        """
        x = x.transpose(1, 2)                  # (B, in_dim, T)
        x = self.conv(x).transpose(1, 2)       # (B, T, hidden)
        out, _ = self.rnn(x)
        frame_emb = self.linear(out)           # (B, T, out_dim)
        frame_emb = nn.functional.normalize(frame_emb, dim=-1)

        # Global prosody (temporal pooling)
        pooled = self.pool(frame_emb.transpose(1, 2)).squeeze(-1)
        global_emb = nn.functional.normalize(pooled, dim=-1)

        return frame_emb, global_emb


# ------------------------------
# Feature extraction helper
# ------------------------------

def extract_features(wav_path, sr=16000, hop_length=256, n_mels=80):
    y, _ = librosa.load(wav_path, sr=sr)
    y, _ = librosa.effects.trim(y, top_db=20)

    # --- Mel-spectrogram ---
    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=1024, hop_length=hop_length,
        n_mels=n_mels, fmin=0, fmax=sr // 2
    )
    mel = librosa.power_to_db(mel).T  # (T, n_mels)

    # --- F0 and energy ---
    y64 = y.astype(np.float64)
    _f0, t = pw.harvest(y64, sr, f0_floor=50.0, f0_ceil=600.0)
    f0 = pw.stonemask(y64, _f0, t, sr)
    f0 = np.log(f0 + 1e-6)
    f0 = (f0 - np.mean(f0)) / (np.std(f0) + 1e-6)
    f0 = np.interp(np.linspace(0, len(f0), mel.shape[0]), np.arange(len(f0)), f0)

    energy = librosa.feature.rms(y=y, frame_length=1024, hop_length=hop_length).flatten()
    energy = np.interp(np.linspace(0, len(energy), mel.shape[0]), np.arange(len(energy)), energy)
    energy = np.log(energy + 1e-6)
    energy = (energy - np.mean(energy)) / (np.std(energy) + 1e-6)

    feats = np.concatenate([mel, f0[:, None], energy[:, None]], axis=1)
    return torch.from_numpy(feats).float().unsqueeze(0)  # (1, T, D)


# ------------------------------
# Example usage
# ------------------------------

if __name__ == "__main__":
    model = ProsodyEncoder(out_dim=128)
    audio_path = "sample.wav"

    feats = extract_features(audio_path)

    with torch.no_grad():
        frame_emb, global_emb = model(feats)

    print("Frame-level embedding shape:", frame_emb.shape)   # (1, T, 128)
    print("Global embedding shape:", global_emb.shape)       # (1, 128)
    print("Preview global embedding (first 10 dims):", global_emb[0, :10])
