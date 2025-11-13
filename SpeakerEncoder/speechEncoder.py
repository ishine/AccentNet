import torch
from speechbrain.pretrained import EncoderClassifier
from tqdm import tqdm
import torchaudio

class SpeakerEncoder:
    def __init__(self, model_name="speechbrain/spkrec-ecapa-voxceleb", device="cpu"):
        self.classifier = EncoderClassifier.from_hparams(source=model_name, run_opts={"device": device})
        self.device = device

    def extract_embedding(self, audio_path):
        signal, fs = torchaudio.load(audio_path)
        if signal.shape[0] > 1:
            signal = torch.mean(signal, dim=0, keepdim=True)
        signal = signal.squeeze()
        if fs != 16000:
            resampler = torchaudio.transforms.Resample(fs, 16000)
            signal = resampler(signal)
        with torch.no_grad():
            embedding = self.classifier.encode_batch(signal.unsqueeze(0).to(self.device))
        return embedding.squeeze().cpu().numpy()
    
    def get_file_list(self, data_list_path):
        file_list = []
        with open(data_list_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    file_list.append((parts[0], parts[1]))  # (audio_path, speaker_id)
        return file_list
    
    def get_embedddings_for_speakers(self, file_list):
        speaker_embeddings = {}
        for audio_path, speaker_id in tqdm(file_list, desc="Extracting Embeddings"):
            embedding = self.extract_embedding(audio_path)
            if speaker_id not in speaker_embeddings:
                speaker_embeddings[speaker_id] = []
            speaker_embeddings[speaker_id].append(embedding)
        return speaker_embeddings