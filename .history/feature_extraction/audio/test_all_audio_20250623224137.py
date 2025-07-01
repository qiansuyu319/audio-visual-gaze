import subprocess
import numpy as np
import os
import tempfile
import soundfile as sf

def check_feature(path, expected_shape=None, name=""):
    assert os.path.exists(path), f"{name} {path} not found"
    arr = np.load(path)
    if expected_shape:
        assert arr.shape == expected_shape, f"{name} shape mismatch: {arr.shape} vs {expected_shape}"
    assert not np.isnan(arr).any(), f"{name} feature contains NaN"
    assert not np.isinf(arr).any(), f"{name} feature contains Inf"
    assert arr.std() > 0, f"{name} feature is constant or all zeros"
    print(f"{name} {path} check passed.")

if __name__ == '__main__':
    # eGeMAPS
    print('Testing extract_egemaps.py...')
    with tempfile.TemporaryDirectory() as tmpdir:
        wav_path = os.path.join(tmpdir, 'test.wav')
        out_path = os.path.join(tmpdir, 'egemaps.npy')
        sf.write(wav_path, np.random.randn(16000), 16000)
        subprocess.run(['python', 'extract_egemaps.py', '--input_wav', wav_path, '--output_npy', out_path], cwd='.', check=True)
        check_feature(out_path, expected_shape=(1, 88), name="eGeMAPS")

    # wav2vec2.0
    print('Testing extract_wav2vec.py...')
    with tempfile.TemporaryDirectory() as tmpdir:
        wav_path = os.path.join(tmpdir, 'test.wav')
        out_path = os.path.join(tmpdir, 'wav2vec.npy')
        sf.write(wav_path, np.random.randn(16000), 16000)
        subprocess.run(['python', 'extract_wav2vec.py', '--input_wav', wav_path, '--output_npy', out_path], cwd='.', check=True)
        arr = np.load(out_path)
        assert arr.ndim in [1, 2], f"wav2vec2.0 feature ndim unexpected: {arr.ndim}"
        assert arr.shape[-1] == 768, f"wav2vec2.0 feature last dim should be 768, got {arr.shape}"
        assert not np.isnan(arr).any(), "wav2vec2.0 feature contains NaN"
        assert not np.isinf(arr).any(), "wav2vec2.0 feature contains Inf"
        assert arr.std() > 0, "wav2vec2.0 feature is constant or all zeros"
        print(f"wav2vec2.0 {out_path} check passed.")

    # preprocess_audio
    print('Testing preprocess_audio.py...')
    with tempfile.TemporaryDirectory() as tmpdir:
        wav_path = os.path.join(tmpdir, 'test.wav')
        out_dir = os.path.join(tmpdir, 'chunks')
        sf.write(wav_path, np.random.randn(16000 * 3), 16000)
        subprocess.run(['python', 'preprocess_audio.py', '--input_wav', wav_path, '--output_dir', out_dir, '--chunk_length', '1.0'], cwd='.', check=True)
        files = os.listdir(out_dir)
        wav_files = [f for f in files if f.endswith('.wav')]
        assert len(wav_files) >= 2, "Audio was not split into chunks as expected"
        print(f"preprocess_audio.py produced {len(wav_files)} chunks, check passed.")

    print('All audio feature extraction tests passed!') 