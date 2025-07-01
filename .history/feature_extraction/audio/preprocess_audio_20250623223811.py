# Placeholder for audio segmentation/normalization

import argparse
import soundfile as sf
import numpy as np
import os

def preprocess_audio(input_wav, output_dir, chunk_length=2.0):
    os.makedirs(output_dir, exist_ok=True)
    data, sr = sf.read(input_wav)
    max_val = np.max(np.abs(data))
    if max_val > 0:
        data = data / max_val
    total_len = len(data) / sr
    n_chunks = int(np.ceil(total_len / chunk_length))
    for i in range(n_chunks):
        start = int(i * chunk_length * sr)
        end = int(min((i + 1) * chunk_length * sr, len(data)))
        chunk = data[start:end]
        out_path = os.path.join(output_dir, f'chunk_{i:03d}.wav')
        sf.write(out_path, chunk, sr)
        print(f'Saved {out_path}')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_wav', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--chunk_length', type=float, default=2.0)
    args = parser.parse_args()
    preprocess_audio(args.input_wav, args.output_dir, args.chunk_length)

if __name__ == '__main__':
    main() 