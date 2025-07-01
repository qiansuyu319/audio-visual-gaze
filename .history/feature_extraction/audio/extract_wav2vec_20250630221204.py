# Placeholder for wav2vec2.0 feature extraction

import argparse
import torch
import torchaudio
from transformers import Wav2Vec2Model, Wav2Vec2Processor
import numpy as np
import os

def get_wav2vec_model(model_name='facebook/wav2vec2-base-960h', device=None):
    """
    Initializes and returns a Wav2Vec2.0 model and its processor.
    The model is loaded onto the specified device (GPU if available).

    Args:
        model_name (str): The name of the pre-trained Wav2Vec2.0 model from the Hugging Face model hub.
        device (str, optional): The device to load the model onto ('cuda' or 'cpu'). 
                                If None, it automatically detects GPU availability.

    Returns:
        tuple: A tuple containing the loaded model and processor.
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    processor = Wav2Vec2Processor.from_pretrained(model_name)
    model = Wav2Vec2Model.from_pretrained(model_name).to(device)
    model.eval()
    print(f"Wav2Vec2 model '{model_name}' loaded on device '{device}'.")
    return model, processor

def extract_wav2vec_from_chunk(model, processor, chunk, sr):
    """
    Extracts Wav2Vec2.0 features from a single audio segment.
    This function computes the mean of the last hidden state from the model as the feature representation.

    Args:
        model: The pre-initialized Wav2Vec2.0 model.
        processor: The corresponding model processor.
        chunk (np.ndarray): A 1D numpy array representing the audio signal.
        sr (int): The sample rate of the audio chunk.

    Returns:
        np.ndarray: A numpy array containing the extracted Wav2Vec2.0 features.
    """
    device = model.device
    inputs = processor(chunk, sampling_rate=sr, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        features = model(**inputs).last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
    return features

def sliding_window_extract(input_wav, output_dir, window_size=1.0, stride=0.04, sr=16000, model_name='facebook/wav2vec2-base-960h'):
    """
    Processes an audio file using a sliding window to extract Wav2Vec2.0 features from each window.
    This function handles audio loading, preprocessing, and windowing, and saves features to .npy files.
    """
    os.makedirs(output_dir, exist_ok=True)
    waveform, file_sr = torchaudio.load(input_wav)
    
    # Resample if the sample rate does not match the target
    if file_sr != sr:
        waveform = torchaudio.functional.resample(waveform, file_sr, sr)

    # Convert to mono by averaging channels
    if waveform.ndim == 2 and waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0)
    else:
        waveform = waveform.squeeze()
    waveform = waveform.numpy()

    win_len = int(window_size * sr)
    hop_len = int(stride * sr)
    total_len = len(waveform)
    
    if total_len < win_len:
        print(f"Warning: Audio duration is less than the window size for {input_wav}. Skipping.")
        return
        
    n_windows = 1 + int(np.floor((total_len - win_len) / hop_len))
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, processor = get_wav2vec_model(model_name, device=device)
    
    print(f"Extracting Wav2Vec2.0 features for {n_windows} windows from {os.path.basename(input_wav)}...")
    for i in range(n_windows):
        start = i * hop_len
        end = start + win_len
        chunk = waveform[start:end]
        if len(chunk) < win_len:
            continue
        
        features = extract_wav2vec_from_chunk(model, processor, chunk, sr)
        
        out_path = os.path.join(output_dir, f'wav2vec_{i:05d}.npy')
        np.save(out_path, features)
    print(f"Wav2Vec2.0 feature extraction complete. Files saved to {output_dir}")

def main():
    """The main entry point for running the script from the command line."""
    parser = argparse.ArgumentParser(description="Extracts Wav2Vec2.0 features from an audio file using a sliding window approach.")
    parser.add_argument('--input_wav', required=True, help="Path to the input audio file.")
    parser.add_argument('--output_dir', required=False, default='output/audio_feature', help="Directory to save the output .npy files. 推荐放在 output/audio_feature/ (默认: output/audio_feature)")
    parser.add_argument('--window_size', type=float, default=1.0, help='The size of the sliding window in seconds.')
    parser.add_argument('--stride', type=float, default=0.04, help='The step or stride of the sliding window in seconds.')
    parser.add_argument('--sr', type=int, default=16000, help='The target sample rate for audio processing.')
    parser.add_argument('--model_name', type=str, default='facebook/wav2vec2-base-960h', help='The pre-trained Wav2Vec2 model to use.')
    args = parser.parse_args()
    sliding_window_extract(args.input_wav, args.output_dir, args.window_size, args.stride, args.sr, args.model_name)

if __name__ == '__main__':
    main() 