import torch
from torch.utils.data import Dataset
import numpy as np
import pickle
import os

def _pad_or_trim(array, target_len):
    """Pad (with zeros) or trim array along first axis to match target_len."""
    current_len = array.shape[0]
    if current_len == target_len:
        return array
    if current_len > target_len:
        return array[:target_len]
    pad_width = [(0, 0)] * array.ndim
    pad_width[0] = (0, target_len - current_len)
    return np.pad(array, pad_width, mode='constant')

def _quantize_durations(duration, mel_len):
    """Convert float durations to integers whose sum matches mel_len."""
    duration = duration.astype(np.float64)
    total = duration.sum()
    if total <= 0 or mel_len <= 0:
        return np.zeros_like(duration, dtype=np.float32)
    
    scaled = duration / total * mel_len
    base = np.floor(scaled).astype(np.int64)
    residual = scaled - base
    deficit = int(mel_len - base.sum())
    
    if deficit > 0:
        order = np.argsort(-residual)
        if len(order) > 0:
            reps = (deficit + len(order) - 1) // len(order)
            indices = np.tile(order, reps)
            np.add.at(base, indices[:deficit], 1)
    elif deficit < 0:
        order = np.argsort(residual)
        if len(order) > 0:
            need = -deficit
            reps = (need + len(order) - 1) // len(order)
            indices = np.tile(order, reps)
            removed = 0
            for idx in indices:
                if base[idx] > 0:
                    base[idx] -= 1
                    removed += 1
                    if removed >= need:
                        break
    
    return base.astype(np.float32)
class AccentConversionDataset(Dataset):
    def __init__(self, metadata_file, embeddings_dir):
        """
        Args:
            metadata_file: Text file with source_id | target_id per line
            embeddings_dir: Directory with pre-extracted embeddings
        """
        self.embeddings_dir = embeddings_dir
        self.pairs = []
        
        with open(metadata_file, 'r') as f:
            for line in f:
                stripped = line.strip()
                if not stripped or stripped.startswith('#'):
                    continue
                parts = [p.strip() for p in stripped.split('|')]
                if len(parts) != 3:
                    raise ValueError(
                        f"Metadata line should contain 3 '|' separated fields: {line}"
                    )
                source_id, target_id, sentence_id = parts
                self.pairs.append((source_id, target_id, sentence_id))
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        source_id, target_id, sentence_id = self.pairs[idx]
        
        # Load source embeddings
        source_path = os.path.join(self.embeddings_dir, source_id)
        content_emb = np.load(source_path + f'/{sentence_id}_content.npy')  # [seq_len, 768]
        speaker_emb = np.load(source_path + f'/{sentence_id}_speaker.npy')  #
        
        # Load prosody from source (retain original rhythm)
        with open(source_path + f'/{sentence_id}_prosody.pkl', 'rb') as f:
            prosody = pickle.load(f)

        # Load target data
        target_path = os.path.join(self.embeddings_dir, target_id)
        accent_emb = np.load(target_path + f'/{sentence_id}_accent.npy')    # [seq_len, 256]
        mel_target = np.load(target_path + f'/{sentence_id}_mel.npy')  # [mel_len, 80]

        # Align sequence lengths
        seq_len = content_emb.shape[0]
        accent_emb = _pad_or_trim(accent_emb, seq_len)
        duration = _pad_or_trim(prosody['duration'], seq_len)
        mel_len = mel_target.shape[0]
        pitch = _pad_or_trim(prosody['pitch'], mel_len)
        energy = _pad_or_trim(prosody['energy'], mel_len)
        duration = _quantize_durations(duration, mel_len)

        # print(seq_len, accent_emb.shape[0])
        # print(pitch.shape[0])
        
        return {
            'content_emb': torch.from_numpy(content_emb).float(),
            'speaker_emb': torch.from_numpy(speaker_emb).float(),
            'accent_emb': torch.from_numpy(accent_emb).float(),
            'pitch': torch.from_numpy(pitch).float(),
            'energy': torch.from_numpy(energy).float(),
            'duration': torch.from_numpy(duration).float(),
            'mel_target': torch.from_numpy(mel_target).float(),
        }
