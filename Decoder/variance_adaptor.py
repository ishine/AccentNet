import torch
import torch.nn as nn
import torch.nn.functional as F

class AccentVarianceAdaptor(nn.Module):
    def __init__(self, hidden_dim=256, num_pitch_bins=256, num_energy_bins=256):
        super().__init__()
        self.pitch_embedding = nn.Embedding(num_pitch_bins, hidden_dim)
        self.energy_embedding = nn.Embedding(num_energy_bins, hidden_dim)
        self.dropout = nn.Dropout(0.1)
    
    def quantize(self, values, num_bins, v_min=0, v_max=1):
        """Quantize continuous values to discrete bins."""
        if values.dim() == 3:
            values = values.squeeze(-1)
        
        values = torch.clamp(values, v_min, v_max)
        boundaries = torch.linspace(v_min, v_max, num_bins, device=values.device)
        bins = torch.bucketize(values, boundaries)
        bins = torch.clamp(bins, 0, num_bins - 1)
        return bins
    
    def forward(self, encoder_output, pitch_target=None, energy_target=None, 
                duration_target=None):
        output = encoder_output
        expanded_lengths = None

        # Length regulation must happen before injecting mel-level features.
        if duration_target is not None:
            output, expanded_lengths = self.length_regulate(output, duration_target)
        else:
            expanded_lengths = torch.full(
                (encoder_output.size(0),),
                encoder_output.size(1),
                dtype=torch.long,
                device=encoder_output.device,
            )
        pad_mask = self._lengths_to_mask(expanded_lengths, output.size(1), output.device)
        
        # Add pitch (mel-level)
        if pitch_target is not None:
            pitch_bins = self.quantize(
                pitch_target,
                self.pitch_embedding.num_embeddings,
                50,
                400
            )
            pitch_emb = self.pitch_embedding(pitch_bins)
            pitch_emb = self._match_time_dimension(pitch_emb, output.size(1))
            pitch_emb = pitch_emb.masked_fill(pad_mask.unsqueeze(-1), 0.0)
            output = output + self.dropout(pitch_emb)
        
        # Add energy (mel-level)
        if energy_target is not None:
            energy_bins = self.quantize(
                energy_target,
                self.energy_embedding.num_embeddings,
                0,
                1
            )
            energy_emb = self.energy_embedding(energy_bins)
            energy_emb = self._match_time_dimension(energy_emb, output.size(1))
            energy_emb = energy_emb.masked_fill(pad_mask.unsqueeze(-1), 0.0)
            output = output + self.dropout(energy_emb)
        
        return output, expanded_lengths
    
    def length_regulate(self, encoder_output, duration_target):
        batch_size, seq_len, hidden_dim = encoder_output.shape
        durations = torch.round(duration_target).long()
        positive = durations > 0
        durations = torch.where(positive, durations.clamp(min=1), torch.zeros_like(durations))
        
        output_list = []
        lengths = durations.sum(dim=1)
        for i in range(batch_size):
            repeated = torch.repeat_interleave(
                encoder_output[i],
                durations[i],
                dim=0
            )
            output_list.append(repeated)
        
        # Pad to max length
        max_len = max(o.shape[0] for o in output_list)
        padded = encoder_output.new_zeros(batch_size, max_len, hidden_dim)
        
        for i, o in enumerate(output_list):
            padded[i, : o.shape[0]] = o
        
        return padded, lengths

    def _match_time_dimension(self, embedding, target_length):
        """Ensure embeddings match decoder input length."""
        emb_len = embedding.size(1)
        if emb_len == target_length:
            return embedding
        if emb_len > target_length:
            return embedding[:, :target_length, :]
        pad_amount = target_length - emb_len
        padded = F.pad(embedding.transpose(1, 2), (0, pad_amount))
        return padded.transpose(1, 2)

    def _lengths_to_mask(self, lengths, max_len, device):
        range_tensor = torch.arange(max_len, device=device).unsqueeze(0)
        return range_tensor >= lengths.unsqueeze(1)
