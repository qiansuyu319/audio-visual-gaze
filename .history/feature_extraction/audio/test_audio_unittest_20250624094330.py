import unittest
import os
import numpy as np
import soundfile as sf
import tempfile
import shutil
import subprocess
import json

AUDIO_DIR = os.path.dirname(__file__)
PROCESS_SCRIPT = os.path.join(AUDIO_DIR, 'process_audio.py')

def create_wav(path, sr=16000, duration=1.0, n_channels=1, value=0.5):
    """Helper function to create a dummy WAV file."""
    samples = int(sr * duration)
    if n_channels == 1:
        data = (np.ones(samples) * value).astype(np.float32)
    else:
        data = (np.ones((samples, n_channels)) * value).astype(np.float32)
    sf.write(path, data, sr)

class TestProcessAudio(unittest.TestCase):
    def setUp(self):
        """Create a temporary directory for test artifacts."""
        self.tmpdir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Remove the temporary directory after tests."""
        shutil.rmtree(self.tmpdir)

    def test_normal_case_all_features(self):
        """Tests standard processing on a mono WAV file, extracting all features."""
        wav_path = os.path.join(self.tmpdir, 'mono.wav')
        out_dir = os.path.join(self.tmpdir, 'output_normal')
        create_wav(wav_path, sr=16000, duration=2.5)

        cmd = ['python', PROCESS_SCRIPT, '--input_audio', wav_path, '--output_dir', out_dir, '--cpu']
        subprocess.run(cmd, check=True, capture_output=True)

        # Check for meta.json and its content
        meta_path = os.path.join(out_dir, 'meta.json')
        self.assertTrue(os.path.exists(meta_path))
        with open(meta_path, 'r') as f:
            meta = json.load(f)
        
        num_windows = meta['params']['total_windows']
        self.assertGreater(num_windows, 0)

        # Check for feature files
        egemaps_files = [f for f in os.listdir(out_dir) if f.endswith('_egemaps.npy')]
        wav2vec_files = [f for f in os.listdir(out_dir) if f.endswith('_wav2vec.npy')]
        self.assertEqual(len(egemaps_files), num_windows)
        self.assertEqual(len(wav2vec_files), num_windows)

        # Check feature shapes
        egemaps_feat = np.load(os.path.join(out_dir, egemaps_files[0]))
        wav2vec_feat = np.load(os.path.join(out_dir, wav2vec_files[0]))
        self.assertEqual(egemaps_feat.shape, (88,))
        self.assertEqual(wav2vec_feat.shape, (768,))

    def test_multichannel_and_save_wav(self):
        """Tests processing of a stereo WAV and the --save_wav_windows flag."""
        wav_path = os.path.join(self.tmpdir, 'stereo.wav')
        out_dir = os.path.join(self.tmpdir, 'output_multichannel')
        create_wav(wav_path, sr=22050, duration=2.5, n_channels=2)

        cmd = ['python', PROCESS_SCRIPT, '--input_audio', wav_path, '--output_dir', out_dir, '--save_wav_windows', '--cpu']
        subprocess.run(cmd, check=True, capture_output=True)

        meta_path = os.path.join(out_dir, 'meta.json')
        self.assertTrue(os.path.exists(meta_path))
        with open(meta_path, 'r') as f:
            meta = json.load(f)
        
        num_windows = meta['params']['total_windows']
        self.assertGreater(num_windows, 0)
        self.assertEqual(meta['params']['target_sr'], 16000)

        # Check that wav windows were saved
        wav_files = [f for f in os.listdir(out_dir) if f.endswith('.wav')]
        self.assertEqual(len(wav_files), num_windows)

        # Check one saved wav file
        data, sr = sf.read(os.path.join(out_dir, wav_files[0]))
        self.assertEqual(sr, 16000)
        self.assertEqual(data.ndim, 1) # Should be mono

    def test_short_audio(self):
        """Tests processing of an audio file shorter than the window size."""
        wav_path = os.path.join(self.tmpdir, 'short.wav')
        out_dir = os.path.join(self.tmpdir, 'output_short')
        create_wav(wav_path, sr=16000, duration=0.5) # Shorter than 1.0s window

        cmd = ['python', PROCESS_SCRIPT, '--input_audio', wav_path, '--output_dir', out_dir, '--cpu']
        subprocess.run(cmd, check=True, capture_output=True)

        meta_path = os.path.join(out_dir, 'meta.json')
        self.assertTrue(os.path.exists(meta_path))
        with open(meta_path, 'r') as f:
            meta = json.load(f)
        
        self.assertEqual(meta['params']['total_windows'], 1)

    def test_single_feature_extraction(self):
        """Tests extracting only one type of feature."""
        wav_path = os.path.join(self.tmpdir, 'mono_single.wav')
        out_dir = os.path.join(self.tmpdir, 'output_single_feat')
        create_wav(wav_path, sr=16000, duration=2.5)

        cmd = ['python', PROCESS_SCRIPT, '--input_audio', wav_path, '--output_dir', out_dir, '--feature_types', 'wav2vec', '--cpu']
        subprocess.run(cmd, check=True, capture_output=True)
        
        # Check that ONLY wav2vec files were created
        egemaps_files = [f for f in os.listdir(out_dir) if f.endswith('_egemaps.npy')]
        wav2vec_files = [f for f in os.listdir(out_dir) if f.endswith('_wav2vec.npy')]
        self.assertEqual(len(egemaps_files), 0)
        self.assertGreater(len(wav2vec_files), 0)

        # Check meta
        meta_path = os.path.join(out_dir, 'meta.json')
        with open(meta_path, 'r') as f:
            meta = json.load(f)
        self.assertEqual(meta['params']['features_extracted'], ['wav2vec'])

if __name__ == '__main__':
    unittest.main() 