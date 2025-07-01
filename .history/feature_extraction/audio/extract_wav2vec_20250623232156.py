# Placeholder for wav2vec2.0 feature extraction

import argparse
import torch
import torchaudio
from transformers import Wav2Vec2Model, Wav2Vec2Processor
import numpy as np
import os

def sliding_window_extract(input_wav, output_dir, window_size=1.0, stride=0.04, sr=16000, model_name='facebook/wav2vec2-base-960h'):
    os.makedirs(output_dir, exist_ok=True)
    waveform, file_sr = torchaudio.load(input_wav)
    if file_sr != sr:
        waveform = torchaudio.functional.resample(waveform, file_sr, sr)
    waveform = waveform.squeeze().numpy()
    win_len = int(window_size * sr)
    hop_len = int(stride * sr)
    processor = Wav2Vec2Processor.from_pretrained(model_name)
    model = Wav2Vec2Model.from_pretrained(model_name)
    model.eval()
    n_windows = int(np.floor((len(waveform) - win_len) / hop_len) + 1)
    print(f"waveform shape: {waveform.shape}, win_len: {win_len}, hop_len: {hop_len}, n_windows: {n_windows}")
    for i in range(n_windows):
        start = i * hop_len
        end = start + win_len
        chunk = waveform[start:end]
        if len(chunk) < win_len:
            break
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