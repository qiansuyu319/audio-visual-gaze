import unittest
import numpy as np
import os
import tempfile
import soundfile as sf
import subprocess

AUDIO_DIR = os.path.dirname(os.path.abspath(__file__))

class TestAudioFeatureExtraction(unittest.TestCase):
    def check_feature(self, path, expected_shape=None, name=""):
        self.assertTrue(os.path.exists(path), f"{name} {path} not found")
        arr = np.load(path)
        if expected_shape:
            self.assertEqual(arr.shape, expected_shape, f"{name} shape mismatch: {arr.shape} vs {expected_shape}")
        self.assertFalse(np.isnan(arr).any(), f"{name} feature contains NaN")
        self.assertFalse(np.isinf(arr).any(), f"{name} feature contains Inf")
        self.assertGreater(arr.std(), 0, f"{name} feature is constant or all zeros")

    def test_extract_egemaps(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            wav_path = os.path.join(tmpdir, 'test.wav')
            out_dir = os.path.join(tmpdir, 'egemaps')
            os.makedirs(out_dir, exist_ok=True)
            sf.write(wav_path, np.random.randn(16000), 16000)
            subprocess.run(['python', os.path.join(AUDIO_DIR, 'extract_egemaps.py'), '--input_wav', wav_path, '--output_dir', out_dir], check=True)
            files = [f for f in os.listdir(out_dir) if f.endswith('.npy')]
            self.assertGreater(len(files), 0, "No egemaps features extracted")
            self.check_feature(os.path.join(out_dir, files[0]), expected_shape=(1, 88), name="eGeMAPS")

    def test_extract_wav2vec(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            wav_path = os.path.join(tmpdir, 'test.wav')
            out_dir = os.path.join(tmpdir, 'wav2vec')
            os.makedirs(out_dir, exist_ok=True)
            sf.write(wav_path, np.random.randn(16000), 16000)
            subprocess.run(['python', os.path.join(AUDIO_DIR, 'extract_wav2vec.py'), '--input_wav', wav_path, '--output_dir', out_dir], check=True)
            files = [f for f in os.listdir(out_dir) if f.endswith('.npy')]
            self.assertGreater(len(files), 0, "No wav2vec features extracted")
            arr = np.load(os.path.join(out_dir, files[0]))
            self.assertIn(arr.ndim, [1, 2], f"wav2vec2.0 feature ndim unexpected: {arr.ndim}")
            self.assertEqual(arr.shape[-1], 768, f"wav2vec2.0 feature last dim should be 768, got {arr.shape}")
            self.assertFalse(np.isnan(arr).any(), "wav2vec2.0 feature contains NaN")
            self.assertFalse(np.isinf(arr).any(), "wav2vec2.0 feature contains Inf")
            self.assertGreater(arr.std(), 0, "wav2vec2.0 feature is constant or all zeros")

    def test_preprocess_audio(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            wav_path = os.path.join(tmpdir, 'test.wav')
            out_dir = os.path.join(tmpdir, 'chunks')
            sf.write(wav_path, np.random.randn(16000 * 3), 16000)
            subprocess.run(['python', os.path.join(AUDIO_DIR, 'preprocess_audio.py'), '--input_wav', wav_path, '--output_dir', out_dir, '--chunk_length', '1.0'], check=True)
            files = os.listdir(out_dir)
            wav_files = [f for f in files if f.endswith('.wav')]
            self.assertGreaterEqual(len(wav_files), 2, "Audio was not split into chunks as expected")

if __name__ == '__main__':
    unittest.main() 