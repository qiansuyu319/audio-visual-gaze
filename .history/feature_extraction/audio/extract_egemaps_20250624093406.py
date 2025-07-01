import argparse
import os
import opensmile
import soundfile as sf
import numpy as np
import torch
import torchaudio

def get_egemaps_model():
    """Initializes and returns an openSMILE model for eGeMAPS feature extraction."""
    return opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.Functionals,
    )

def extract_egemaps_from_chunk(model, chunk, sr):
    """Extracts eGeMAPS features from a single audio chunk."""
    features = model.process_signal(chunk, sr)
    return features.values.squeeze()

def sliding_window_extract(input_wav, output_dir, window_size=1.0, stride=0.04, sr=16000):
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        data, file_sr = torchaudio.load(input_wav)
        data = data.squeeze().numpy()
    except Exception:
        data, file_sr = sf.read(input_wav)

    if data.ndim == 2:
        data = data.mean(axis=1)
    
    if file_sr != sr:
        print(f"Resampling from {file_sr} Hz to {sr} Hz...")
        data = torchaudio.functional.resample(torch.from_numpy(data), orig_freq=file_sr, new_freq=sr).numpy()

    smile = get_egemaps_model()

    win_len = int(window_size * sr)
    hop_len = int(stride * sr)
    
    if len(data) < win_len:
        print(f"Audio {input_wav} too short for window size, skipping.")
        return

    n_windows = 1 + int(np.floor((len(data) - win_len) / hop_len))
    print(f"Extracting eGeMAPS for {n_windows} windows...")
    for i in range(n_windows):
        start = i * hop_len
        end = start + win_len
        chunk = data[start:end]
        if len(chunk) < win_len:
            continue
        
        features = extract_egemaps_from_chunk(smile, chunk, sr)
        
        out_path = os.path.join(output_dir, f'egemaps_{i:05d}.npy')
        np.save(out_path, features)
    print(f"eGeMAPS features saved to {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Extracts eGeMAPS features from an audio file using a sliding window.")
    parser.add_argument('--input_wav', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--window_size', type=float, default=1.0, help='Window size in seconds')
    parser.add_argument('--stride', type=float, default=0.04, help='Stride in seconds')
    parser.add_argument('--sr', type=int, default=16000, help='Sample rate')
    args = parser.parse_args()
    sliding_window_extract(args.input_wav, args.output_dir, args.window_size, args.stride, args.sr)

if __name__ == '__main__':
    main() 