import torch
import torch.nn.functional as F

def collate_fn_variable_length(batch):
    """Collate variable-length sequences with padding."""
    
    # Convert torch.Size to int
    max_content = max(int(s['content_emb'].shape[0]) for s in batch)
    max_accent = max(int(s['accent_emb'].shape[0]) for s in batch)
    max_mel = max(int(s['mel_target'].shape[0]) for s in batch)
    
    result = {}
    
    # Content embedding [seq, 768] -> [batch, max_seq, 768]
    result['content_emb'] = torch.stack([
        F.pad(s['content_emb'], (0, 0, 0, max_content - int(s['content_emb'].shape[0])))
        for s in batch
    ])
    
    # Speaker embedding  -> [batch, 192] (no padding)
    result['speaker_emb'] = torch.stack([s['speaker_emb'] for s in batch])
    
    # Accent embedding [seq, 256] -> [batch, max_seq, 256]
    result['accent_emb'] = torch.stack([
        F.pad(s['accent_emb'], (0, 0, 0, max_accent - int(s['accent_emb'].shape[0])))
        for s in batch
    ])
    
    # Pitch [mel] -> [batch, max_mel]
    result['pitch'] = torch.stack([
        F.pad(s['pitch'], (0, max_mel - int(s['pitch'].shape[0])))
        for s in batch
    ])
    
    # Energy [mel] -> [batch, max_mel]
    result['energy'] = torch.stack([
        F.pad(s['energy'], (0, max_mel - int(s['energy'].shape[0])))
        for s in batch
    ])
    
    # Duration [seq] -> [batch, max_seq]
    result['duration'] = torch.stack([
        F.pad(s['duration'], (0, max_content - int(s['duration'].shape[0])))
        for s in batch
    ])
    
    # Mel target [mel, 80] -> [batch, max_mel, 80]
    result['mel_target'] = torch.stack([
        F.pad(s['mel_target'], (0, 0, 0, max_mel - int(s['mel_target'].shape[0])))
        for s in batch
    ])
    
    return result
