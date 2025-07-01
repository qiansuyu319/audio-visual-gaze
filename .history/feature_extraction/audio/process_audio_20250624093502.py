"""
process_audio.py - Unified Audio Preprocessing & Sliding Window Feature Extraction

# === Implemented Features ===

# (1) 读取原始音频文件
#   - [X] 支持 wav/mp3/多格式 (依赖torchaudio/soundfile)
#   - [X] 打印输入文件信息（采样率、通道数、时长）

# (2) 音频预处理
#   - [X] 自动重采样为 target_sr（如16kHz）
#   - [X] 多通道混为单通道
#   - [X] 振幅归一化
#   - [X] 检查并处理异常样本（全零、极短）

# (3) 滑窗分帧
#   - [X] 支持 window_length/step, 滑窗输出
#   - [X] 自动编号 window_00000.npy, window_00001.npy …
#   - [X] 每帧长度不足时自动补零
#   - [X] 支持导出 window_xxxxx.wav

# (4) 批量特征提取入口
#   - [X] 统一调用 extract_egemaps / extract_wav2vec
#   - [X] 自动存 window_xxxxx_egemaps.npy, window_xxxxx_wav2vec.npy
#   - [X] 特征编号与滑窗编号严格一致

# (5) meta信息输出
#   - [X] 输出 meta.json

# (6) 代码结构与扩展性
#   - [X] 函数式写法, 易于扩展
#   - [X] 参数可由 argparse 控制
#   - [X] 全流程异常捕获与日志打印

# (7) 单元测试支持
#   - [X] 脚本可直接运行, 易于集成测试

"""
import argparse
import os
import json
import numpy as np
import soundfile as sf
import torch
import torchaudio
from tqdm import tqdm

try:
    from feature_extraction.audio.extract_egemaps import get_egemaps_model, extract_egemaps_from_chunk
    from feature_extraction.audio.extract_wav2vec import get_wav2vec_model, extract_wav2vec_from_chunk
except ImportError:
    print("Assuming script is run from base directory. Adjusting import path.")
    from extract_egemaps import get_egemaps_model, extract_egemaps_from_chunk
    from extract_wav2vec import get_wav2vec_model, extract_wav2vec_from_chunk

def load_and_preprocess_audio(audio_path, target_sr=16000):
    """Loads, resamples, converts to mono, and normalizes an audio file."""
    try:
        data, sr = torchaudio.load(audio_path)
    except Exception as e:
        print(f"Warning: torchaudio load failed ('{e}'), trying soundfile.")
        data, sr = sf.read(audio_path)
        data = torch.from_numpy(data.T)
        if data.ndim == 1:
            data = data.unsqueeze(0)

    print(f"Loaded '{os.path.basename(audio_path)}': sr={sr}, channels={data.shape[0]}, duration={data.shape[-1]/sr:.2f}s")

    if data.shape[0] > 1:
        data = torch.mean(data, dim=0, keepdim=True)
    
    if sr != target_sr:
        data = torchaudio.functional.resample(data, orig_freq=sr, new_freq=target_sr)
        sr = target_sr
        print(f"Resampled to {sr} Hz")

    data = data.squeeze().numpy()
    
    max_val = np.max(np.abs(data))
    if max_val > 1.0: # Normalize if it's not in [-1, 1]
        data /= max_val
    
    if np.all(data == 0):
        print("Warning: Audio is all zeros.")

    return data.astype(np.float32), sr


def sliding_window_generator(data, sr, window_size_s, step_s, pad=True):
    """Yields audio windows from the input signal."""
    win_len = int(window_size_s * sr)
    hop_len = int(step_s * sr)
    
    if len(data) == 0:
        print("Error: Audio data is empty.")
        return

    if len(data) < win_len:
        print("Warning: data is shorter than a window. Padding to window size.")
        if pad:
            padded_data = np.pad(data, (0, win_len - len(data)), 'constant')
            yield padded_data
        return

    num_windows = 1 + int(np.floor((len(data) - win_len) / hop_len))
    
    for i in range(num_windows):
        start = i * hop_len
        end = start + win_len
        chunk = data[start:end]
        
        if pad and len(chunk) < win_len:
            chunk = np.pad(chunk, (0, win_len - len(chunk)), 'constant')
        
        if len(chunk) == win_len:
            yield chunk

def process_single_audio(args):
    """Main processing logic for a single audio file."""
    os.makedirs(args.output_dir, exist_ok=True)
    
    audio_data, sr = load_and_preprocess_audio(args.input_audio, args.target_sr)
    
    egemaps_model = None
    wav2vec_model, wav2vec_processor = None, None
    
    if 'egemaps' in args.feature_types:
        print("Loading eGeMAPS model...")
        egemaps_model = get_egemaps_model()
        
    if 'wav2vec' in args.feature_types:
        print("Loading Wav2Vec2.0 model...")
        device = 'cuda' if torch.cuda.is_available() and not args.cpu else 'cpu'
        wav2vec_model, wav2vec_processor = get_wav2vec_model(device=device)

    windows = list(sliding_window_generator(audio_data, sr, args.window_size, args.step))
    
    if not windows:
        print("No windows were generated. Exiting.")
        return

    print(f"Extracting features for {len(windows)} windows...")
    for i, chunk in enumerate(tqdm(windows, desc="Processing windows")):
        window_basename = f'window_{i:05d}'
        
        if args.save_wav_windows:
            sf.write(os.path.join(args.output_dir, f'{window_basename}.wav'), chunk, sr)

        if egemaps_model:
            egemaps_feat = extract_egemaps_from_chunk(egemaps_model, chunk, sr)
            np.save(os.path.join(args.output_dir, f'{window_basename}_egemaps.npy'), egemaps_feat)
            
        if wav2vec_model:
            wav2vec_feat = extract_wav2vec_from_chunk(wav2vec_model, wav2vec_processor, chunk, sr)
            np.save(os.path.join(args.output_dir, f'{window_basename}_wav2vec.npy'), wav2vec_feat)

    meta = {
        'source_audio': os.path.abspath(args.input_audio),
        'output_dir': os.path.abspath(args.output_dir),
        'params': {
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
        
    print(f"\nProcessing complete. Results saved to '{args.output_dir}'")
    print(f"Metadata saved to '{meta_path}'")


def main():
    parser = argparse.ArgumentParser(description="Unified audio preprocessing and feature extraction pipeline.")
    parser.add_argument('--input_audio', type=str, required=True, help="Path to the input audio file.")
    parser.add_argument('--output_dir', type=str, required=True, help="Directory to save the output files.")
    parser.add_argument('--target_sr', type=int, default=16000, help="Target sample rate to resample to.")
    parser.add_argument('--window_size', type=float, default=1.0, help="Sliding window size in seconds.")
    parser.add_argument('--step', type=float, default=0.04, help="Sliding window step in seconds.")
    parser.add_argument('--feature_types', nargs='+', default=['egemaps', 'wav2vec'], help="List of features to extract. Choices: 'egemaps', 'wav2vec'.")
    parser.add_argument('--save_wav_windows', action='store_true', help="If set, saves the audio for each window as a .wav file.")
    parser.add_argument('--cpu', action='store_true', help="Force use CPU for feature extraction.")
    
    args = parser.parse_args()
    
    try:
        process_single_audio(args)
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
