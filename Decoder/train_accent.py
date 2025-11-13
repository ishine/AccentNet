import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
import argparse
import os
from torch.utils.data import DataLoader
import yaml

# try:
#     from .model import FastSpeech2Accent
#     from .dataset import AccentConversionDataset
#     from .utils import collate_fn_variable_length
#     from .manifest_dataset import ManifestAccentPairDataset
# except ImportError:
from model import FastSpeech2Accent
from dataset import AccentConversionDataset
from utils import collate_fn_variable_length
from manifest_dataset import ManifestAccentPairDataset

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

class TrainConfig:
    def __init__(self, config_dict):
        for k, v in config_dict.items():
            setattr(self, k, v)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--train_meta', type=str, default='data/train.txt')
    parser.add_argument('--val_meta', type=str, default='data/val.txt')
    parser.add_argument('--embeddings_dir', type=str, default='data/embeddings')
    parser.add_argument('--manifest', type=str, default=None,
                        help='JSONL manifest describing embeddings (optional)')
    parser.add_argument('--source_accent', type=str, default=None,
                        help='Source accent name in manifest mode')
    parser.add_argument('--target_accent', type=str, default=None,
                        help='Target accent name in manifest mode')
    parser.add_argument('--source_dataset', type=str, default=None,
                        help='Optional source dataset filter in manifest mode')
    parser.add_argument('--target_dataset', type=str, default=None,
                        help='Optional target dataset filter in manifest mode')
    parser.add_argument('--source_speaker', type=str, default=None,
                        help='Optional source speaker filter in manifest mode')
    parser.add_argument('--target_speaker', type=str, default=None,
                        help='Optional target speaker filter in manifest mode')
    parser.add_argument('--output_dir', type=str, default='checkpoints')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001)
    args = parser.parse_args()
    
    # Load config
    config_dict = load_config(args.config)
    config = TrainConfig(config_dict)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Model
    model = FastSpeech2Accent(config).to(device)
    
    def _read_utterance_list(path):
        if not path or not os.path.exists(path):
            return None
        ids = []
        with open(path, 'r') as f:
            for line in f:
                stripped = line.strip()
                if not stripped or stripped.startswith('#'):
                    continue
                parts = [p.strip() for p in stripped.split('|')]
                if len(parts) == 3:
                    ids.append(parts[2])
                else:
                    ids.append(stripped)
        return ids

    if args.manifest:
        if not args.source_accent or not args.target_accent:
            raise ValueError("Manifest mode requires --source_accent and --target_accent.")
        dataset_kwargs = dict(
            manifest_path=args.manifest,
            root_dir=args.embeddings_dir,
            source_accent=args.source_accent,
            target_accent=args.target_accent,
            source_dataset=args.source_dataset,
            target_dataset=args.target_dataset,
            source_speaker=args.source_speaker,
            target_speaker=args.target_speaker,
            require_mel=True,
        )
        train_dataset = ManifestAccentPairDataset(
            allowed_utterances=_read_utterance_list(args.train_meta),
            **dataset_kwargs
        )
        val_dataset = ManifestAccentPairDataset(
            allowed_utterances=_read_utterance_list(args.val_meta),
            **dataset_kwargs
        )
    else:
        # Datasets
        train_dataset = AccentConversionDataset(args.train_meta, args.embeddings_dir)
        val_dataset = AccentConversionDataset(args.val_meta, args.embeddings_dir)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn_variable_length)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, collate_fn=collate_fn_variable_length)
    
    # Loss & optimizer
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    
    # Training loop
    for epoch in range(args.epochs):
        # Train
        model.train()
        train_loss = 0
        for batch in train_loader:
            content_emb = batch['content_emb'].to(device)
            speaker_emb = batch['speaker_emb'].to(device)
            accent_emb = batch['accent_emb'].to(device)
            pitch = batch['pitch'].to(device)
            energy = batch['energy'].to(device)
            duration = batch['duration'].to(device)
            mel_target = batch['mel_target'].to(device)
            
            mel_pred, mel_pred_refined = model(
                content_emb, speaker_emb, accent_emb,
                pitch_target=pitch,
                energy_target=energy,
                duration_target=duration
            )
            
            loss = criterion(mel_pred, mel_target) + criterion(mel_pred_refined, mel_target)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validate
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                content_emb = batch['content_emb'].to(device)
                speaker_emb = batch['speaker_emb'].to(device)
                accent_emb = batch['accent_emb'].to(device)
                pitch = batch['pitch'].to(device)
                energy = batch['energy'].to(device)
                duration = batch['duration'].to(device)
                mel_target = batch['mel_target'].to(device)
                
                mel_pred, mel_pred_refined = model(
                    content_emb, speaker_emb, accent_emb,
                    pitch_target=pitch,
                    energy_target=energy,
                    duration_target=duration
                )
                
                loss = criterion(mel_pred, mel_target) + criterion(mel_pred_refined, mel_target)
                val_loss += loss.item()
        
        scheduler.step()
        
        avg_train = train_loss / len(train_loader)
        avg_val = val_loss / len(val_loader)
        print(f"Epoch {epoch+1}/{args.epochs} - Train: {avg_train:.4f}, Val: {avg_val:.4f}")
        
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f"{args.output_dir}/model_epoch_{epoch+1}.pt")

if __name__ == '__main__':
    main()
