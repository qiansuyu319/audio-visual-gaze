import argparse
import os
import opensmile
import soundfile as sf
import numpy as np
import torch
import torchaudio

def get_egemaps_model():
    return opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.Functionals,
    )

def extract_egemaps_from_chunk(model, chunk, sr):
    features = model.process_signal(chunk, sr)
    return features.values.squeeze()

def sliding_window_extract(input_wav, output_dir, window_size=5.0, stride=0.04, sr=16000):
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        data, file_sr = torchaudio.load(input_wav)
        data = data.squeeze().numpy()
    except Exception:
        data, file_sr = sf.read(input_wav)

    # Convert stereo to mono
    if data.ndim == 2:
        data = data.mean(axis=1)
    
    if file_sr != sr:
        print(f"Resampling from {file_sr} Hz to {sr} Hz...")
        data = torchaudio.functional.resample(torch.from_numpy(data), orig_freq=file_sr, new_freq=sr).numpy()

    smile = get_egemaps_model()
    
    win_len = int(window_size * sr)
    hop_len = int(stride * sr)

    total_frames = 1 + int(np.floor((len(data) - win_len) / hop_len))
    if total_frames <= 0:
        print(f"❌ Warning: Audio too short for {input_wav}. Skipped.")
        return

    print(f"🎧 Extracting eGeMAPS from {input_wav} with {total_frames} frames...")

    for i in range(total_frames):
        start = i * hop_len
        end = start + win_len
        if end > len(data):
            break  # Skip incomplete window at the end
        chunk = data[start:end]
        features = extract_egemaps_from_chunk(smile, chunk, sr)
        
        out_path = os.path.join(output_dir, f'egemaps_{i:05d}.npy')
        np.save(out_path, features)

    print(f"✅ Saved {total_frames} feature vectors to {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Extract frame-aligned eGeMAPS using sliding windows")
    parser.add_argument('--input_wav', required=True, help="Path to input .wav file")
    parser.add_argument('--output_dir', required=True, help="Directory to save per-frame .npy features")
    parser.add_argument('--window_size', type=float, default=5.0, help="Sliding window size in seconds (default: 5.0)")
    parser.add_argument('--stride', type=float, default=0.04, help="Stride in seconds (default: 0.04s, matches 25 FPS)")
    parser.add_argument('--sr', type=int, default=16000, help="Target sampling rate (default: 16000Hz)")
    args = parser.parse_args()

    sliding_window_extract(
        input_wav=args.input_wav,
        output_dir=args.output_dir,
        window_size=args.window_size,
        stride=args.stride,
        sr=args.sr
    )

if __name__ == '__main__':
    main()
