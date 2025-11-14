import argparse
import os
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch
import whisper


from identity_encoder import inference as encoder
from identity_encoder.params_model import model_embedding_size as speaker_embedding_size
from mel_synthesizer.inference import Synthesizer
from utils.argutils import print_args
from utils.default_models import ensure_default_models
from signal_vocoder import inference as vocoder

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("-e", "--enc_model_fpath", type=Path,
                        default="saved_models/default/encoder.pt",
                        help="Path to a saved encoder")
    parser.add_argument("-s", "--syn_model_fpath", type=Path,
                        default="saved_models/999/synthesizer.pt",
                        help="Path to a saved synthesizer")
    parser.add_argument("-v", "--voc_model_fpath", type=Path,
                        default="saved_models/999/vocoder.pt",
                        help="Path to a saved vocoder")
    parser.add_argument("--cpu", action="store_true", help=\
        "If True, processing is done on CPU, even when a GPU is available.")
    parser.add_argument("--no_sound", action="store_true", help=\
        "If True, audio won't be played.")
    parser.add_argument("--seed", type=int, default=None, help=\
        "Optional random number seed value to make lab_console deterministic.")
    args = parser.parse_args()
    arg_dict = vars(args)
    print_args(args, parser)

    # Hide GPUs from Pytorch to force CPU processing
    if arg_dict.pop("cpu"):
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    if torch.cuda.is_available():
        device_id = torch.cuda.current_device()
        gpu_properties = torch.cuda.get_device_properties(device_id)
    else:
        print("Using CPU for inference.\n")

    ## Load the models one by one.
    ensure_default_models(Path("saved_models"))
    encoder.load_model(args.enc_model_fpath)
    synthesizer = Synthesizer(args.syn_model_fpath)
    vocoder.load_model(args.voc_model_fpath)
    if args.in_fpath is not None:
        in_fpath = args.in_fpath
        model = whisper.load_model("base")  
        result = model.transcribe(str(in_fpath))
        text = result["text"].strip()
        out_fpath = args.out_fpath
    
        print(f"\nUsing reference voice: {in_fpath}")
        print(f"Synthesizing text: \"{text}\"")
        print(f"Output will be saved to: {out_fpath}\n")
    
        # Load and preprocess reference audio
        preprocessed_wav = encoder.preprocess_wav(in_fpath)
        embed = encoder.embed_utterance(preprocessed_wav)
        print("Created speaker embedding")
    
        # Generate mel spectrogram
        texts = [text]
        embeds = [embed]
        specs = synthesizer.synthesize_spectrograms(texts, embeds)
        spec = specs[0]
        print("Created mel spectrogram")
    
        # Generate waveform
        generated_wav = vocoder.infer_waveform(spec)
        generated_wav = np.pad(generated_wav, (0, synthesizer.sample_rate), mode="constant")
        generated_wav = encoder.preprocess_wav(generated_wav)
    
        # Save to file
        sf.write(out_fpath, generated_wav.astype(np.float32), synthesizer.sample_rate)
        print(f"\n✅ Saved output audio to {out_fpath}\n")
    
        if not args.no_sound:
            import sounddevice as sd
            try:
                sd.stop()
                sd.play(generated_wav, synthesizer.sample_rate)
            except Exception as e:
                print(f"Audio playback skipped: {e}")
    
        exit(0)
    
