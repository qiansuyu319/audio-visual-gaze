# Placeholder for wav2vec2.0 feature extraction

import argparse
import torch
import torchaudio
from transformers import Wav2Vec2Model, Wav2Vec2Processor
import numpy as np
import os
import soundfile as sf
import librosa

def load_and_preprocess(audio_path, target_sr=16000):
    data, orig_sr = sf.read(audio_path)
    # Convert to mono if multi-channel
    if data.ndim == 2:
        print(f"Input audio has {data.shape[1]} channels, converting to mono by averaging channels.")
        data = data.mean(axis=1)
    # Resample if needed
    if orig_sr != target_sr:
        print(f"Resampling from {orig_sr} Hz to {target_sr} Hz...")
        data = librosa.resample(data.astype(float), orig_sr, target_sr)
    # Ensure float32
    data = data.astype(np.float32)
    return data, target_sr

def sliding_window_extract(input_wav, output_dir, window_size=1.0, stride=0.04, sr=16000, model_name='facebook/wav2vec2-base-960h'):
    os.makedirs(output_dir, exist_ok=True)
    data, sr = load_and_preprocess(input_wav, sr)
    win_len = int(window_size * sr)
    hop_len = int(stride * sr)
    n_samples = len(data)
    if n_samples < win_len:
        print(f"Audio {input_wav} too short for window size, skipping.")
        return
    n_windows = 1 + (n_samples - win_len) // hop_len
    print(f"waveform shape: {data.shape}, win_len: {win_len}, hop_len: {hop_len}, n_windows: {n_windows}")
    processor = Wav2Vec2Processor.from_pretrained(model_name)
    model = Wav2Vec2Model.from_pretrained(model_name)
    model.eval()
    for i in range(n_windows):
        start = i * hop_len
        end = start + win_len
        chunk = data[start:end]
        if len(chunk) < win_len:
            print(f"Skipping short chunk at window {i}, length {len(chunk)}")
            continue
        # Ensure input shape is [n_samples,] for processor
        inputs = processor(chunk, sampling_rate=sr, return_tensors="pt", padding=True)
        with torch.no_grad():
            features = model(**inputs).last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
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