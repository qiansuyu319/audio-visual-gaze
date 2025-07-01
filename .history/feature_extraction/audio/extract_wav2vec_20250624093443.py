# Placeholder for wav2vec2.0 feature extraction

import argparse
import torch
import torchaudio
from transformers import Wav2Vec2Model, Wav2Vec2Processor
import numpy as np
import os

def get_wav2vec_model(model_name='facebook/wav2vec2-base-960h', device=None):
    """Initializes and returns a Wav2Vec2.0 model and processor on the specified device."""
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    processor = Wav2Vec2Processor.from_pretrained(model_name)
    model = Wav2Vec2Model.from_pretrained(model_name).to(device)
    model.eval()
    print(f"Wav2Vec2 model loaded on {device}")
    return model, processor

def extract_wav2vec_from_chunk(model, processor, chunk, sr):
    """Extracts Wav2Vec2.0 features from a single audio chunk."""
    device = model.device
    inputs = processor(chunk, sampling_rate=sr, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        features = model(**inputs).last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
    return features

def sliding_window_extract(input_wav, output_dir, window_size=1.0, stride=0.04, sr=16000, model_name='facebook/wav2vec2-base-960h'):
    os.makedirs(output_dir, exist_ok=True)
    waveform, file_sr = torchaudio.load(input_wav)
    if file_sr != sr:
        waveform = torchaudio.functional.resample(waveform, file_sr, sr)
    # 如果是多通道，转为单通道
    if waveform.ndim == 2 and waveform.shape[0] > 1:
        print(f"Input audio has {waveform.shape[0]} channels, converting to mono by averaging channels.")
        waveform = waveform.mean(dim=0)
    else:
        waveform = waveform.squeeze()
    waveform = waveform.numpy()
    win_len = int(window_size * sr)
    hop_len = int(stride * sr)
    total_len = len(waveform)
    if total_len < win_len:
        print(f"Audio {input_wav} too short for window size, skipping.")
        return
    n_windows = int(np.floor((total_len - win_len) / hop_len) + 1)
    print(f"waveform shape: {waveform.shape}, win_len: {win_len}, hop_len: {hop_len}, n_windows: {n_windows}")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, processor = get_wav2vec_model(model_name, device=device)
    
    print(f"Extracting Wav2Vec2.0 for {n_windows} windows...")
    for i in range(n_windows):
        start = i * hop_len
        end = start + win_len
        chunk = waveform[start:end]
        if len(chunk) < win_len:
            continue
        
        features = extract_wav2vec_from_chunk(model, processor, chunk, sr)
        
        out_path = os.path.join(output_dir, f'wav2vec_{i:05d}.npy')
        np.save(out_path, features)
    print(f"Wav2Vec2.0 features saved to {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Extracts Wav2Vec2.0 features from an audio file using a sliding window.")
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