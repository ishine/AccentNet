import torch
import torch.nn as nn

class MultiEncoderInput(nn.Module):
    def __init__(self, content_dim=768, speaker_dim=192, accent_dim=256, hidden_dim=256):
        super(MultiEncoderInput, self).__init__()
        total_dim = content_dim + speaker_dim + accent_dim
        self.projection = nn.Linear(total_dim, hidden_dim)
    
    def forward(self, content_emb, speaker_emb, accent_emb):
        """
        Args:
            content_emb: [batch, seq_len, 768]
            speaker_emb: [batch, 192]  ← No seq_len!
            accent_emb: [batch, seq_len, 256]
        
        Returns:
            fused: [batch, seq_len, output_dim]
        """
        
        if speaker_emb.dim() == 1:
            speaker_emb = speaker_emb.unsqueeze(0)
        seq_len = content_emb.shape[1]
        
        # ✓ KEY FIX: Expand speaker from [batch, 192] to [batch, seq_len, 192]
        speaker_expanded = speaker_emb.unsqueeze(1).expand(-1, seq_len, -1)
        
        # Now all have seq_len dimension - can concatenate
        fused = torch.cat([content_emb, speaker_expanded, accent_emb], dim=-1)
        valid_mask = (content_emb.abs().sum(dim=-1, keepdim=True) > 0).float()
        fused = fused * valid_mask
        
        # Project to output dimension
        fused = self.projection(fused)
        
        return fused
