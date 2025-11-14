import os, glob, random
import torch, torchaudio
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
import numpy as np

TARGET_SR = 16000

#extract data into pairs <audio, accent label>
def scan_by_accent(root):
    pairs = []  # (wav_path, accent_label)
    for accent in sorted(os.listdir(root)):
        adir = os.path.join(root, accent)
        if not os.path.isdir(adir): continue
        for p in glob.glob(os.path.join(adir, "**/*.wav"), recursive=True):
            pairs.append((p, accent))
    return pairs

def getSpeaker(path, accent):
    parts = path.split(os.sep)
    speaker = ""
    for part in parts:
        if part.startswith("speaker_"):
            speaker = part 

    speaker = accent + "_" + speaker
    return speaker

#TODO: ensure accent diversity
def train_val_test_split(pairs, train=0.8, val=0.1):
    random.shuffle(pairs)
    n = len(pairs) 
    ntr = int(n*train)
    nva = int(n*val)
    return pairs[:ntr], pairs[ntr:ntr+nva], pairs[ntr+nva:]

def pool_mean_std(feats):
    # feats: [1, T', 768] -> [1536]
    if feats.dim() == 3:
        feats = feats[0]
    mu  = feats.mean(dim=0)
    std = feats.std(dim=0)
    return torch.cat([mu, std], dim=0)  # [1536] 

def load_and_resample_16k(path):
    wav, sr = torchaudio.load(path)       # [C, T]
    wav = wav.mean(dim=0, keepdim=True)   # mono [1, T]
    if sr != TARGET_SR:
        wav = torchaudio.functional.resample(wav, sr, TARGET_SR)
    wav = wav / (wav.abs().max() + 1e-8)  # peak-norm
    return wav

def scan_and_split_by_accent(root, val_ratio=0.2, random_state=42):

    train_pairs = []
    val_pairs   = []
    test_pairs  = []

    for accent in sorted(os.listdir(root)):
        accent_dir = os.path.join(root, accent)
        if not os.path.isdir(accent_dir):
            continue

        # get speaker_01 and speaker_02 paths
        spk1_dir = os.path.join(accent_dir, "speaker_01")
        spk2_dir = os.path.join(accent_dir, "speaker_02")

        spk1_wavs = []
        spk2_wavs = []

        if os.path.isdir(spk1_dir):
            spk1_wavs = sorted(glob.glob(os.path.join(spk1_dir, "*.wav")))
        if os.path.isdir(spk2_dir):
            spk2_wavs = sorted(glob.glob(os.path.join(spk2_dir, "*.wav")))

        # Split speaker1 into train + val
        if len(spk1_wavs) > 0:
            tr, va = train_test_split(
                spk1_wavs,
                test_size=val_ratio,
                random_state=random_state
            )
            train_pairs += [(p, accent) for p in tr]
            val_pairs   += [(p, accent) for p in va]

        # Speaker2 files for test
        if len(spk2_wavs) > 0:
            test_pairs += [(p, accent) for p in spk2_wavs]

    return train_pairs, val_pairs, test_pairs

def generate_frame_level_dataset(feature_set, label_set, k = 20):
    data = []
    labels = []
    for i in range(len(feature_set)):
        d, l = generate_frame_level_data(feature_set[i], label_set[i], k)
        data.append(d)
        labels.append(l)
    frame_features = torch.cat(data, dim=0)
    frame_labels = torch.cat(labels, dim=0)
    return frame_features, frame_labels
    
def generate_frame_level_data(feats, label, k = 20):
    if isinstance(feats, np.ndarray):
        feats = torch.from_numpy(feats).float()
    if feats.dim() == 3 and feats.size(0) == 1:
        # Convert [1, T', D] -> [T', D]
        feats = feats.squeeze(0)
    random_indices = np.random.randint(1, len(feats), size=k)
    selected_features = feats[random_indices]
    labels = torch.full(size=(k,), fill_value=label)
    return selected_features, labels 

def near_duplicate_pairs(Xtr, Xte, topk=1, cos_thr=0.99):
    Xt = Xtr #torch.tensor(Xtr, dtype=torch.float32)
    Xe = Xte #torch.tensor(Xte, dtype=torch.float32)
    Xt = F.normalize(Xt, dim=1); Xe = F.normalize(Xe, dim=1)
    sim = Xe @ Xt.T                       
    max_sim, idx = sim.max(dim=1)
    hits = (max_sim.numpy() >= cos_thr).sum()
    print(f"Test items with cosine ≥ {cos_thr}: {hits}")
    return hits    