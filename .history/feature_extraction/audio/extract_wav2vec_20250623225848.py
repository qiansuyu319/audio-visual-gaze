# Placeholder for wav2vec2.0 feature extraction

import argparse
import torch
import torchaudio
from transformers import Wav2Vec2Model, Wav2Vec2Processor
import numpy as np
import os
import librosa

# Sliding window extraction for wav2vec2.0 features
# Supports automatic resampling to target sample rate (default 16kHz)
def sliding_window_extract(input_wav, output_dir, window_size=1.0, stride=0.04, sr=16000, model_name='facebook/wav2vec2-base-960h'):
    os.makedirs(output_dir, exist_ok=True)
    # Load audio file
    waveform, file_sr = torchaudio.load(input_wav)
    # If sample rate does not match, resample using librosa
    if file_sr != sr:
        print(f"Resampling from {file_sr} Hz to {sr} Hz...")
        # torchaudio.load returns (channels, samples), convert to numpy for librosa
        np_wave = waveform.squeeze().numpy()
        np_wave = librosa.resample(np_wave, orig_sr=file_sr, target_sr=sr)
        waveform = torch.from_numpy(np_wave).unsqueeze(0)
    else:
        waveform = waveform.squeeze().numpy()
    # Prepare sliding window parameters
    win_len = int(window_size * sr)
    hop_len = int(stride * sr)
    processor = Wav2Vec2Processor.from_pretrained(model_name)
    model = Wav2Vec2Model.from_pretrained(model_name)
    # If waveform is still torch.Tensor, convert to numpy
    if isinstance(waveform, torch.Tensor):
        waveform = waveform.numpy()
    n_windows = int(np.floor((len(waveform) - win_len) / hop_len) + 1)
    for i in range(n_windows):
        start = i * hop_len
        end = start + win_len
        chunk = waveform[start:end]
        if len(chunk) < win_len:
            break
        # Extract wav2vec2.0 features for this chunk
        inputs = processor(chunk, sampling_rate=sr, return_tensors="pt", padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
        # Use mean pooling over time dimension
        features = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
        out_path = os.path.join(output_dir, f'wav2vec_{i:05d}.npy')
        np.save(out_path, features)
        print(f"Saved {out_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_wav', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--window_size', type=float, default=1.0, help='Window size in seconds')
    parser.add_argument('--stride', type=float, default=0.04, help='Stride in seconds')
    parser.add_argument('--sr', type=int, default=16000, help='Sample rate')
    parser.add_argument('--model_name', type=str, default='facebook/wav2vec2-base-960h', help='Wav2Vec2 model name')
    args = parser.parse_args()
    sliding_window_extract(args.input_wav, args.output_dir, args.window_size, args.stride, args.sr, args.model_name)

if __name__ == '__main__':
    main() 