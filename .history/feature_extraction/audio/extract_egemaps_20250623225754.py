import argparse
import os
import opensmile
import soundfile as sf
import numpy as np
import librosa

def sliding_window_extract(input_wav, output_dir, window_size=1.0, stride=0.04, sr=16000):
    os.makedirs(output_dir, exist_ok=True)
    data, file_sr = sf.read(input_wav)
    if file_sr != sr:
        print(f"Resampling from {file_sr} Hz to {sr} Hz...")
        data = librosa.resample(data.T, orig_sr=file_sr, target_sr=sr).T if data.ndim == 2 else librosa.resample(data, orig_sr=file_sr, target_sr=sr)
    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.Functionals,
    )
    win_len = int(window_size * sr)
    hop_len = int(stride * sr)
    n_windows = int(np.floor((len(data) - win_len) / hop_len) + 1)
    for i in range(n_windows):
        start = i * hop_len
        end = start + win_len
        chunk = data[start:end]
        if len(chunk) < win_len:
            break
        features = smile.process_signal(chunk, sr)
        out_path = os.path.join(output_dir, f'egemaps_{i:05d}.npy')
        np.save(out_path, features.values)
        print(f"Saved {out_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_wav', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--window_size', type=float, default=1.0, help='Window size in seconds')
    parser.add_argument('--stride', type=float, default=0.04, help='Stride in seconds')
    parser.add_argument('--sr', type=int, default=16000, help='Sample rate')
    args = parser.parse_args()
    sliding_window_extract(args.input_wav, args.output_dir, args.window_size, args.stride, args.sr)

if __name__ == '__main__':
    main() 