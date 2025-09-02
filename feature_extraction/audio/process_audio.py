import argparse
import os
import json
import numpy as np
import soundfile as sf
import torch
import torchaudio
from tqdm import tqdm

from feature_extraction.audio.extract_egemaps import get_egemaps_model, extract_egemaps_from_chunk
from feature_extraction.audio.extract_wav2vec import get_wav2vec_model, extract_wav2vec_from_chunk

def load_and_preprocess_audio(audio_path, target_sr=16000):
    try:
        data, sr = torchaudio.load(audio_path)
    except Exception as e:
        print(f"Fallback to soundfile for {audio_path}: {e}")
        data, sr = sf.read(audio_path)
        data = torch.from_numpy(data.T)
        if data.ndim == 1:
            data = data.unsqueeze(0)

    if data.shape[0] > 1:
        data = torch.mean(data, dim=0, keepdim=True)

    if sr != target_sr:
        data = torchaudio.functional.resample(data, orig_freq=sr, new_freq=target_sr)
        sr = target_sr

    data = data.squeeze().numpy()
    max_val = np.max(np.abs(data))
    if max_val > 1.0:
        data /= max_val

    return data.astype(np.float32), sr

def sliding_window_generator(data, sr, window_size_s, step_s, pad=True):
    win_len = int(window_size_s * sr)
    hop_len = int(step_s * sr)

    if len(data) < win_len and pad:
        padded = np.pad(data, (0, win_len - len(data)), 'constant')
        yield padded
        return

    num_windows = 1 + int(np.floor((len(data) - win_len) / hop_len))
    for i in range(num_windows):
        start = i * hop_len
        end = start + win_len
        chunk = data[start:end]
        if pad and len(chunk) < win_len:
            chunk = np.pad(chunk, (0, win_len - len(chunk)), 'constant')
        yield chunk

def process_single_audio(args):
    os.makedirs(args.output_dir, exist_ok=True)
    audio_data, sr = load_and_preprocess_audio(args.input_audio, args.target_sr)

    # Load models
    use_egemaps = 'egemaps' in args.feature_types
    use_wav2vec = 'wav2vec' in args.feature_types

    egemaps_model = get_egemaps_model() if use_egemaps else None
    wav2vec_model, wav2vec_processor = (get_wav2vec_model(device='cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
                                        if use_wav2vec else (None, None))

    windows = list(sliding_window_generator(audio_data, sr, args.window_size, args.step))
    if not windows:
        print("No windows generated.")
        return

    base = os.path.splitext(os.path.basename(args.input_audio))[0]

    # Prepare subfolders
    egemaps_out_dir = os.path.join(args.output_dir, 'egemaps')
    wav2vec_out_dir = os.path.join(args.output_dir, 'wav2vec')
    os.makedirs(egemaps_out_dir, exist_ok=True)
    os.makedirs(wav2vec_out_dir, exist_ok=True)

    # Extract features and save
    if use_egemaps:
        egemaps_feats = [extract_egemaps_from_chunk(egemaps_model, chunk, sr) for chunk in tqdm(windows, desc="eGeMAPS")]
        out_path = os.path.join(egemaps_out_dir, f'{base}.npy')
        np.save(out_path, np.array(egemaps_feats))
        print(f"✅ Saved eGeMAPS to {out_path}")

    if use_wav2vec:
        wav2vec_feats = [extract_wav2vec_from_chunk(wav2vec_model, wav2vec_processor, chunk, sr) for chunk in tqdm(windows, desc="Wav2Vec")]
        out_path = os.path.join(wav2vec_out_dir, f'{base}.npy')
        np.save(out_path, np.array(wav2vec_feats))
        print(f"✅ Saved Wav2Vec to {out_path}")

    # Save metadata
    meta = {
        'audio': args.input_audio,
        'sr': sr,
        'windows': len(windows),
        'features': args.feature_types,
        'duration': len(audio_data) / sr
    }
    meta_path = os.path.join(args.output_dir, f'{base}_meta.json')
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_audio', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--target_sr', type=int, default=16000)
    parser.add_argument('--window_size', type=float, default=5.0)
    parser.add_argument('--step', type=float, default=0.04)
    parser.add_argument('--feature_types', nargs='+', default=['egemaps', 'wav2vec'])
    parser.add_argument('--cpu', action='store_true')
    args = parser.parse_args()

    try:
        process_single_audio(args)
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
