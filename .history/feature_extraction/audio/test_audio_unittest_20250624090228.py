import unittest
import os
import numpy as np
import soundfile as sf
import tempfile
import shutil
import subprocess

AUDIO_DIR = os.path.dirname(__file__)

def create_wav(path, sr=16000, duration=1.0, n_channels=1, value=0.5):
    samples = int(sr * duration)
    if n_channels == 1:
        data = np.ones(samples) * value
    else:
        data = np.ones((samples, n_channels)) * value
    sf.write(path, data, sr)

class TestAudioFeatureExtractionAll(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_extract_egemaps_normal(self):
        wav_path = os.path.join(self.tmpdir, 'mono.wav')
        create_wav(wav_path, sr=16000, duration=2.0, n_channels=1)
        out_dir = os.path.join(self.tmpdir, 'egemaps')
        os.makedirs(out_dir, exist_ok=True)
        subprocess.run(['python', os.path.join(AUDIO_DIR, 'extract_egemaps.py'), '--input_wav', wav_path, '--output_dir', out_dir], check=True)
        files = [f for f in os.listdir(out_dir) if f.endswith('.npy')]
        self.assertGreater(len(files), 0)
        arr = np.load(os.path.join(out_dir, files[0]))
        self.assertEqual(arr.shape, (1, 88))

    def test_extract_egemaps_multichannel(self):
        wav_path = os.path.join(self.tmpdir, 'stereo.wav')
        create_wav(wav_path, sr=16000, duration=2.0, n_channels=2)
        out_dir = os.path.join(self.tmpdir, 'egemaps2')
        os.makedirs(out_dir, exist_ok=True)
        subprocess.run(['python', os.path.join(AUDIO_DIR, 'extract_egemaps.py'), '--input_wav', wav_path, '--output_dir', out_dir], check=True)
        files = [f for f in os.listdir(out_dir) if f.endswith('.npy')]
        self.assertGreater(len(files), 0)

    def test_extract_egemaps_short(self):
        wav_path = os.path.join(self.tmpdir, 'short.wav')
        create_wav(wav_path, sr=16000, duration=0.2, n_channels=1)
        out_dir = os.path.join(self.tmpdir, 'egemaps_short')
        os.makedirs(out_dir, exist_ok=True)
        subprocess.run(['python', os.path.join(AUDIO_DIR, 'extract_egemaps.py'), '--input_wav', wav_path, '--output_dir', out_dir], check=True)
        files = [f for f in os.listdir(out_dir) if f.endswith('.npy')]
        

    def test_extract_wav2vec_normal(self):
        wav_path = os.path.join(self.tmpdir, 'mono.wav')
        create_wav(wav_path, sr=16000, duration=2.0, n_channels=1)
        out_dir = os.path.join(self.tmpdir, 'wav2vec')
        os.makedirs(out_dir, exist_ok=True)
        subprocess.run(['python', os.path.join(AUDIO_DIR, 'extract_wav2vec.py'), '--input_wav', wav_path, '--output_dir', out_dir], check=True)
        files = [f for f in os.listdir(out_dir) if f.endswith('.npy')]
        self.assertGreater(len(files), 0)
        arr = np.load(os.path.join(out_dir, files[0]))
        self.assertIn(arr.shape, [(1, 768), (768,)])

    def test_extract_wav2vec_multichannel(self):
        wav_path = os.path.join(self.tmpdir, 'stereo.wav')
        create_wav(wav_path, sr=16000, duration=2.0, n_channels=2)
        out_dir = os.path.join(self.tmpdir, 'wav2vec2')
        os.makedirs(out_dir, exist_ok=True)
        subprocess.run(['python', os.path.join(AUDIO_DIR, 'extract_wav2vec.py'), '--input_wav', wav_path, '--output_dir', out_dir], check=True)
        files = [f for f in os.listdir(out_dir) if f.endswith('.npy')]
        self.assertGreater(len(files), 0)

    def test_extract_wav2vec_short(self):
        wav_path = os.path.join(self.tmpdir, 'short.wav')
        create_wav(wav_path, sr=16000, duration=0.2, n_channels=1)
        out_dir = os.path.join(self.tmpdir, 'wav2vec_short')
        os.makedirs(out_dir, exist_ok=True)
        subprocess.run(['python', os.path.join(AUDIO_DIR, 'extract_wav2vec.py'), '--input_wav', wav_path, '--output_dir', out_dir], check=True)
        files = [f for f in os.listdir(out_dir) if f.endswith('.npy')]
        # 可能为0

    def test_preprocess_audio_normal(self):
        wav_path = os.path.join(self.tmpdir, 'mono.wav')
        create_wav(wav_path, sr=16000, duration=2.0, n_channels=1)
        out_dir = os.path.join(self.tmpdir, 'chunks')
        os.makedirs(out_dir, exist_ok=True)
        subprocess.run(['python', os.path.join(AUDIO_DIR, 'preprocess_audio.py'), '--input_wav', wav_path, '--output_dir', out_dir, '--chunk_length', '1.0'], check=True)
        files = [f for f in os.listdir(out_dir) if f.endswith('.wav')]
        self.assertGreater(len(files), 0)

    def test_preprocess_audio_multichannel(self):
        wav_path = os.path.join(self.tmpdir, 'stereo.wav')
        create_wav(wav_path, sr=16000, duration=2.0, n_channels=2)
        out_dir = os.path.join(self.tmpdir, 'chunks2')
        os.makedirs(out_dir, exist_ok=True)
        subprocess.run(['python', os.path.join(AUDIO_DIR, 'preprocess_audio.py'), '--input_wav', wav_path, '--output_dir', out_dir, '--chunk_length', '1.0'], check=True)
        files = [f for f in os.listdir(out_dir) if f.endswith('.wav')]
        self.assertGreater(len(files), 0)

    def test_preprocess_audio_short(self):
        wav_path = os.path.join(self.tmpdir, 'short.wav')
        create_wav(wav_path, sr=16000, duration=0.2, n_channels=1)
        out_dir = os.path.join(self.tmpdir, 'chunks_short')
        os.makedirs(out_dir, exist_ok=True)
        subprocess.run(['python', os.path.join(AUDIO_DIR, 'preprocess_audio.py'), '--input_wav', wav_path, '--output_dir', out_dir, '--chunk_length', '1.0'], check=True)
        files = [f for f in os.listdir(out_dir) if f.endswith('.wav')]
        

if __name__ == '__main__':
    unittest.main() 