# Accent Conversion
Accent Morphing Lab is a restructured take on the classic SV2TTS pipeline for accent transfer. It keeps the original research stack but reorganizes the codebase into three clearly named subsystems:

- **identity_encoder** – GE2E-based speaker embedding network.
- **mel_synthesizer** – Tacotron implementation that conditions on embeddings.
- **signal_vocoder** – WaveRNN vocoder for waveform reconstruction.

Fine-tuning two parallel stacks on accent-specific speech (e.g., British vs. American English) allows the system to keep the speaker identity while forcing the generated audio into the target accent.

## Research Roots
| URL | Component | Title | Implementation |
| --- | --- | --- | --- |
| [1806.04558](https://arxiv.org/pdf/1806.04558.pdf) | SV2TTS | Transfer Learning from Speaker Verification to Multispeaker Text-To-Speech Synthesis | Adapted here |
| [1802.08435](https://arxiv.org/pdf/1802.08435.pdf) | WaveRNN | Efficient Neural Audio Synthesis | [fatchord/WaveRNN](https://github.com/fatchord/WaveRNN) |
| [1703.10135](https://arxiv.org/pdf/1703.10135.pdf) | Tacotron | Tacotron: Towards End-to-End Speech Synthesis | [fatchord/WaveRNN](https://github.com/fatchord/WaveRNN) |
| [1710.10467](https://arxiv.org/pdf/1710.10467.pdf) | GE2E | Generalized End-To-End Loss for Speaker Verification | Adapted here |

## Getting Started
1. Python ≥ 3.7 recommended (virtualenv optional).
2. Install [ffmpeg](https://ffmpeg.org/download.html#get-packages).
3. Install [PyTorch](https://pytorch.org/get-started/locally/) for your platform (GPU optional).
4. Install Python dependencies: `pip install -r requirements.txt`.

### Pretrained Models (optional)
Grab the checkpoints from the original release: https://drive.google.com/drive/folders/1hIvW0P2Vj07qsEJ7OTgobfUz6Rr9UgeT?usp=sharing and drop them into `saved_models/`.

### Curating Accent Data
- Mozilla Common Voice is a great starting point: https://www.kaggle.com/datasets/mozillaorg/common-voice
- Place curated accent-specific clips under `accent_corpora/AccentDatasetTrain` and `accent_corpora/AccentDatasetTest` before running the preprocessing pipelines.


1. **identity_encoder** ingests raw audio and produces normalized embeddings using a single-direction LSTM with GE2E loss.
2. **mel_synthesizer** (Tacotron) conditions on embeddings + text to emit mel spectrograms.
3. **signal_vocoder** (WaveRNN) upsamples mels into high-quality waveforms.

To build paired accent models, run the preprocess/train scripts per subsystem:
- `identity_encoder_preprocess.py` → `identity_encoder_train.py`
- `mel_synth_preprocess_audio.py`, `mel_synth_preprocess_embeddings.py` → `mel_synth_train.py`
- `signal_vocoder_preprocess.py` → `signal_vocoder_train.py`

Each script exposes CLI arguments for data locations, run IDs, and checkpoint cadence.

## Audio Samples
- [American source 1](./converted_samples/american1.wav) → [British render](./converted_samples/british1.wav)
- [American source 2](./converted_samples/american2.wav) → [British render](./converted_samples/british2.wav)
- [British source](./converted_samples/british3.wav) → [American render](./converted_samples/american3.wav)

## Notes
- The repository intentionally keeps training/inference logic close to the research papers while reorganizing names and structure for clarity.
- You can plug any speech-to-text front end and second-pass TTS back end for a full transcription + accent translation pipeline if needed.
