import argparse
import os
import opensmile
import soundfile as sf
import numpy as np
import torch
import torchaudio

def get_egemaps_model():
    """
    Initializes and returns the openSMILE model for eGeMAPS feature extraction.
    This function sets up the standardized eGeMAPSv02 feature set.
    """
    return opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.Functionals,
    )

def extract_egemaps_from_chunk(model, chunk, sr):
    """
    Extracts eGeMAPS features from a single audio segment.

    Args:
        model: The pre-initialized openSMILE model.
        chunk (np.ndarray): A 1D numpy array representing the audio signal.
        sr (int): The sample rate of the audio chunk.

    Returns:
        np.ndarray: A numpy array containing the extracted eGeMAPS features.
    """
    features = model.process_signal(chunk, sr)
    return features.values.squeeze()

def sliding_window_extract(input_wav, output_dir, window_size=1.0, stride=0.04, sr=16000):
    """
    Processes an audio file by applying a sliding window and extracting eGeMAPS features from each window.
    This function handles audio loading, resampling, and windowing, then saves the features as .npy files.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        data, file_sr = torchaudio.load(input_wav)
        data = data.squeeze().numpy()
    except Exception:
        # Fallback for formats not supported by torchaudio but by soundfile
        data, file_sr = sf.read(input_wav)

    # Convert to mono by averaging channels if necessary
    if data.ndim == 2:
        data = data.mean(axis=1)
    
    # Resample to the target sample rate if necessary
    if file_sr != sr:
        print(f"Resampling from {file_sr} Hz to {sr} Hz...")
        data = torchaudio.functional.resample(torch.from_numpy(data), orig_freq=file_sr, new_freq=sr).numpy()

    smile = get_egemaps_model()

    win_len = int(window_size * sr)
    hop_len = int(stride * sr)
    
    if len(data) < win_len:
        print(f"Warning: Audio duration is less than the window size for {input_wav}. Skipping.")
        return

    n_windows = 1 + int(np.floor((len(data) - win_len) / hop_len))
    print(f"Extracting eGeMAPS features for {n_windows} windows from {os.path.basename(input_wav)}...")
    for i in range(n_windows):
        start = i * hop_len
        end = start + win_len
        chunk = data[start:end]
        if len(chunk) < win_len:
            continue
        
        features = extract_egemaps_from_chunk(smile, chunk, sr)
        
        out_path = os.path.join(output_dir, f'egemaps_{i:05d}.npy')
        np.save(out_path, features)
    print(f"eGeMAPS feature extraction complete. Files saved to {output_dir}")

def main():
    """The main entry point for running the script from the command line."""
    parser = argparse.ArgumentParser(description="Extracts eGeMAPS features from an audio file using a sliding window approach.")
    parser.add_argument('--input_wav', required=True, help="Path to the input audio file.")
    parser.add_argument('--output_dir', required=False, default='output/audio_feature', help="Directory to save the output .npy files. 推荐放在 output/audio_feature/ (默认: output/audio_feature)")
    parser.add_argument('--window_size', type=float, default=1.0, help='The size of the sliding window in seconds.')
    parser.add_argument('--stride', type=float, default=0.04, help='The step or stride of the sliding window in seconds.')
    parser.add_argument('--sr', type=int, default=16000, help='The target sample rate for audio processing.')
    args = parser.parse_args()
    sliding_window_extract(args.input_wav, args.output_dir, args.window_size, args.stride, args.sr)

if __name__ == '__main__':
    main() 