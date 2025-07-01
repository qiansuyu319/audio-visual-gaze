"""
This script provides a unified pipeline for audio processing and feature extraction.
It is designed to ingest raw audio files, apply a series of preprocessing steps,
and then extract multiple types of features using a synchronized sliding window approach.

The pipeline performs the following key steps:
1.  **Audio Loading**: Supports various audio formats and provides initial metadata.
2.  **Preprocessing**: Converts audio to a standardized format by resampling to a
    target sample rate, converting to mono, and normalizing amplitude.
3.  **Sliding Window Analysis**: Segments the audio into overlapping windows to
    prepare it for time-based feature extraction.
4.  **Feature Extraction**: Extracts specified features (e.g., eGeMAPS, Wav2Vec2.0)
    for each window.
5.  **Serialization**: Saves the extracted features, windowed audio (optional),
    and comprehensive metadata to an output directory.


import argparse
import os
import json
import numpy as np
import soundfile as sf
import torch
import torchaudio
from tqdm import tqdm

try:
    # Assumes the script is run from a context where this path is valid
    from feature_extraction.audio.extract_egemaps import get_egemaps_model, extract_egemaps_from_chunk
    from feature_extraction.audio.extract_wav2vec import get_wav2vec_model, extract_wav2vec_from_chunk
except ImportError:
    # Fallback for running the script directly from its own directory
    print("Assuming script is run from its local directory. Adjusting import path.")
    from extract_egemaps import get_egemaps_model, extract_egemaps_from_chunk
    from extract_wav2vec import get_wav2vec_model, extract_wav2vec_from_chunk

def load_and_preprocess_audio(audio_path, target_sr=16000):
    """
    Loads an audio file, preprocesses it, and returns the signal and sample rate.
    Preprocessing includes conversion to mono, resampling, and amplitude normalization.
    Args:
        audio_path (str): The path to the input audio file.
        target_sr (int): The desired sample rate for the output signal.
    Returns:
        tuple: A tuple containing the preprocessed audio signal as a NumPy array
               and its sample rate as an integer.
    """
    try:
        data, sr = torchaudio.load(audio_path)
    except Exception as e:
        print(f"Warning: torchaudio load failed ('{e}'), attempting fallback with soundfile.")
        data, sr = sf.read(audio_path)
        data = torch.from_numpy(data.T)
        if data.ndim == 1:
            data = data.unsqueeze(0)

    print(f"Loaded '{os.path.basename(audio_path)}': sr={sr}, channels={data.shape[0]}, duration={data.shape[-1]/sr:.2f}s")

    # Convert to mono by averaging channels if necessary
    if data.shape[0] > 1:
        data = torch.mean(data, dim=0, keepdim=True)
    
    # Resample to the target sample rate
    if sr != target_sr:
        data = torchaudio.functional.resample(data, orig_freq=sr, new_freq=target_sr)
        sr = target_sr
        print(f"Resampled signal to {sr} Hz")

    data = data.squeeze().numpy()
    
    # Normalize amplitude to the range [-1, 1] if it exceeds this range
    max_val = np.max(np.abs(data))
    if max_val > 1.0:
        data /= max_val
    
    if np.all(data == 0):
        print("Warning: The audio signal is completely silent (all zeros).")

    return data.astype(np.float32), sr


def sliding_window_generator(data, sr, window_size_s, step_s, pad=True):
    """
    Generates audio windows from a signal using a sliding window.
    Args:
        data (np.ndarray): The 1D audio signal.
        sr (int): The sample rate of the signal.
        window_size_s (float): The length of each window in seconds.
        step_s (float): The step size (stride) of the window in seconds.
        pad (bool): If True, pads the final window with zeros if it is shorter
                    than the specified window size.
    Yields:
        np.ndarray: An audio chunk of size `window_size_s * sr`.
    """
    win_len = int(window_size_s * sr)
    hop_len = int(step_s * sr)
    
    if len(data) == 0:
        print("Error: Input audio data is empty. Cannot generate windows.")
        return

    # Handle cases where the audio is shorter than one window
    if len(data) < win_len:
        print("Warning: Audio data is shorter than one window. Padding to window size.")
        if pad:
            padded_data = np.pad(data, (0, win_len - len(data)), 'constant')
            yield padded_data
        return

    num_windows = 1 + int(np.floor((len(data) - win_len) / hop_len))
    
    for i in range(num_windows):
        start = i * hop_len
        end = start + win_len
        chunk = data[start:end]
        
        # Pad the last chunk if it's shorter than the window length
        if pad and len(chunk) < win_len:
            chunk = np.pad(chunk, (0, win_len - len(chunk)), 'constant')
        
        if len(chunk) == win_len:
            yield chunk

def process_single_audio(args):
    """
    Orchestrates the main processing pipeline for a single audio file.
    This includes loading, preprocessing, windowing, feature extraction, and saving results.
    """
    os.makedirs(args.output_dir, exist_ok=True)
    
    audio_data, sr = load_and_preprocess_audio(args.input_audio, args.target_sr)
    
    # Initialize models based on selected feature types
    egemaps_model = None
    wav2vec_model, wav2vec_processor = None, None
    
    if 'egemaps' in args.feature_types:
        print("Initializing eGeMAPS model...")
        egemaps_model = get_egemaps_model()
        
    if 'wav2vec' in args.feature_types:
        print("Initializing Wav2Vec2.0 model...")
        device = 'cuda' if torch.cuda.is_available() and not args.cpu else 'cpu'
        wav2vec_model, wav2vec_processor = get_wav2vec_model(device=device)

    windows = list(sliding_window_generator(audio_data, sr, args.window_size, args.step))
    
    if not windows:
        print("No windows were generated from the audio file. Exiting.")
        return

    print(f"Extracting features for {len(windows)} windows...")
    for i, chunk in enumerate(tqdm(windows, desc="Processing windows")):
        window_basename = f'window_{i:05d}'
        
        # Optionally save the raw audio for each window
        if args.save_wav_windows:
            sf.write(os.path.join(args.output_dir, f'{window_basename}.wav'), chunk, sr)

        # Extract and save features for the current window
        if egemaps_model:
            egemaps_feat = extract_egemaps_from_chunk(egemaps_model, chunk, sr)
            np.save(os.path.join(args.output_dir, f'{window_basename}_egemaps.npy'), egemaps_feat)
            
        if wav2vec_model:
            wav2vec_feat = extract_wav2vec_from_chunk(wav2vec_model, wav2vec_processor, chunk, sr)
            np.save(os.path.join(args.output_dir, f'{window_basename}_wav2vec.npy'), wav2vec_feat)

    # Serialize metadata for the entire process
    meta = {
        'source_audio': os.path.abspath(args.input_audio),
        'output_dir': os.path.abspath(args.output_dir),
        'processing_parameters': {
            'target_sr': sr,
            'window_size_s': args.window_size,
            'step_s': args.step,
            'total_windows': len(windows),
            'source_duration_s': len(audio_data) / sr,
            'features_extracted': args.feature_types
        }
    }
    meta_path = os.path.join(args.output_dir, 'meta.json')
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=4)
        
    print(f"\nProcessing complete. Results have been saved to '{args.output_dir}'")
    print(f"Metadata summarizing the process has been saved to '{meta_path}'")


def main():
    """Parses command-line arguments and initiates the audio processing pipeline."""
    parser = argparse.ArgumentParser(description="A unified pipeline for audio preprocessing and feature extraction.")
    parser.add_argument('--input_audio', type=str, required=True, help="Path to the source audio file.")
    parser.add_argument('--output_dir', type=str, required=True, help="Directory to store the output files.")
    parser.add_argument('--target_sr', type=int, default=16000, help="Target sample rate for audio resampling.")
    parser.add_argument('--window_size', type=float, default=1.0, help="Duration of the sliding window in seconds.")
    parser.add_argument('--step', type=float, default=0.04, help="Step size (stride) of the sliding window in seconds.")
    parser.add_argument('--feature_types', nargs='+', default=['egemaps', 'wav2vec'], help="A list of features to extract. Available: 'egemaps', 'wav2vec'.")
    parser.add_argument('--save_wav_windows', action='store_true', help="If specified, saves the audio content of each window as a .wav file.")
    parser.add_argument('--cpu', action='store_true', help="Forces the use of CPU for feature extraction, even if a GPU is available.")
    
    args = parser.parse_args()
    
    try:
        process_single_audio(args)
    except Exception as e:
        print(f"\nAn unexpected error occurred during processing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
