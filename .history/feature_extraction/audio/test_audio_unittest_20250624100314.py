import unittest
import os
import numpy as np
import soundfile as sf
import tempfile
import shutil
import subprocess
import json

# Define the absolute path to the directory containing the audio scripts
AUDIO_DIR = os.path.dirname(__file__)
# Define the absolute path to the main processing script to be tested
PROCESS_SCRIPT = os.path.join(AUDIO_DIR, 'process_audio.py')

def create_wav(path, sr=16000, duration=1.0, n_channels=1, value=0.5):
    """
    A helper function to generate a dummy WAV file for testing purposes.

    Args:
        path (str): The file path to save the generated WAV file.
        sr (int): The sample rate of the audio.
        duration (float): The duration of the audio in seconds.
        n_channels (int): The number of audio channels (1 for mono, 2 for stereo).
        value (float): The constant amplitude value for the signal.
    """
    samples = int(sr * duration)
    if n_channels == 1:
        data = (np.ones(samples) * value).astype(np.float32)
    else:
        data = (np.ones((samples, n_channels)) * value).astype(np.float32)
    sf.write(path, data, sr)

class TestProcessAudio(unittest.TestCase):
    """A test suite for the unified `process_audio.py` script."""
    
    def setUp(self):
        """Initializes a temporary directory before each test is run."""
        self.tmpdir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Cleans up and removes the temporary directory after each test."""
        shutil.rmtree(self.tmpdir)

    def test_normal_case_all_features(self):
        """
        Validates the standard pipeline execution on a mono audio file.
        This test checks for the correct generation of all default feature types
        (eGeMAPS, Wav2Vec2.0) and verifies the metadata and output file shapes.
        """
        wav_path = os.path.join(self.tmpdir, 'mono.wav')
        out_dir = os.path.join(self.tmpdir, 'output_normal')
        create_wav(wav_path, sr=16000, duration=2.5)

        cmd = ['python', PROCESS_SCRIPT, '--input_audio', wav_path, '--output_dir', out_dir, '--cpu']
        subprocess.run(cmd, check=True, capture_output=True)

        # Verify that the metadata file was created and contains correct information
        meta_path = os.path.join(out_dir, 'meta.json')
        self.assertTrue(os.path.exists(meta_path), "meta.json was not created.")
        with open(meta_path, 'r') as f:
            meta = json.load(f)
        
        num_windows = meta['processing_parameters']['total_windows']
        self.assertGreater(num_windows, 0, "No windows were processed.")

        # Verify that the correct number of feature files were created
        egemaps_files = [f for f in os.listdir(out_dir) if f.endswith('_egemaps.npy')]
        wav2vec_files = [f for f in os.listdir(out_dir) if f.endswith('_wav2vec.npy')]
        self.assertEqual(len(egemaps_files), num_windows)
        self.assertEqual(len(wav2vec_files), num_windows)

        # Verify the dimensionality of the extracted features
        egemaps_feat = np.load(os.path.join(out_dir, egemaps_files[0]))
        wav2vec_feat = np.load(os.path.join(out_dir, wav2vec_files[0]))
        self.assertEqual(egemaps_feat.shape, (88,), "eGeMAPS feature shape is incorrect.")
        self.assertEqual(wav2vec_feat.shape, (768,), "Wav2Vec2.0 feature shape is incorrect.")

    def test_multichannel_and_save_wav(self):
        """
        Validates the pipeline's handling of multichannel (stereo) audio and the
        --save_wav_windows functionality. It checks for correct resampling,
        conversion to mono, and the creation of windowed WAV files.
        """
        wav_path = os.path.join(self.tmpdir, 'stereo.wav')
        out_dir = os.path.join(self.tmpdir, 'output_multichannel')
        create_wav(wav_path, sr=22050, duration=2.5, n_channels=2)

        cmd = ['python', PROCESS_SCRIPT, '--input_audio', wav_path, '--output_dir', out_dir, '--save_wav_windows', '--cpu']
        subprocess.run(cmd, check=True, capture_output=True)

        meta_path = os.path.join(out_dir, 'meta.json')
        self.assertTrue(os.path.exists(meta_path))
        with open(meta_path, 'r') as f:
            meta = json.load(f)
        
        num_windows = meta['processing_parameters']['total_windows']
        self.assertGreater(num_windows, 0)
        self.assertEqual(meta['processing_parameters']['target_sr'], 16000, "Sample rate was not correctly updated in metadata.")

        # Verify that the windowed .wav files were created
        wav_files = [f for f in os.listdir(out_dir) if f.endswith('.wav')]
        self.assertEqual(len(wav_files), num_windows, "Number of saved WAV windows does not match expected count.")

        # Inspect a saved WAV file to ensure it was processed correctly
        data, sr = sf.read(os.path.join(out_dir, wav_files[0]))
        self.assertEqual(sr, 16000, "Saved WAV window has incorrect sample rate.")
        self.assertEqual(data.ndim, 1, "Saved WAV window is not mono.")

    def test_short_audio_padding(self):
        """
        Tests the pipeline's behavior with an audio file shorter than a single
        window. This validates that the system correctly pads the audio to produce
        exactly one window.
        """
        wav_path = os.path.join(self.tmpdir, 'short.wav')
        out_dir = os.path.join(self.tmpdir, 'output_short')
        create_wav(wav_path, sr=16000, duration=0.5) # Duration is less than the default 1.0s window

        cmd = ['python', PROCESS_SCRIPT, '--input_audio', wav_path, '--output_dir', out_dir, '--cpu']
        subprocess.run(cmd, check=True, capture_output=True)

        meta_path = os.path.join(out_dir, 'meta.json')
        self.assertTrue(os.path.exists(meta_path))
        with open(meta_path, 'r') as f:
            meta = json.load(f)
        
        self.assertEqual(meta['processing_parameters']['total_windows'], 1, "Expected exactly one window for short audio.")

    def test_single_feature_extraction(self):
        """
        Tests the --feature_types argument by requesting only a single feature type.
        This verifies that the pipeline correctly extracts only the specified
        features and logs this in the metadata.
        """
        wav_path = os.path.join(self.tmpdir, 'mono_single.wav')
        out_dir = os.path.join(self.tmpdir, 'output_single_feat')
        create_wav(wav_path, sr=16000, duration=2.5)

        cmd = ['python', PROCESS_SCRIPT, '--input_audio', wav_path, '--output_dir', out_dir, '--feature_types', 'wav2vec', '--cpu']
        subprocess.run(cmd, check=True, capture_output=True)
        
        # Verify that only the requested feature files were created
        egemaps_files = [f for f in os.listdir(out_dir) if f.endswith('_egemaps.npy')]
        wav2vec_files = [f for f in os.listdir(out_dir) if f.endswith('_wav2vec.npy')]
        self.assertEqual(len(egemaps_files), 0, "eGeMAPS files were created when they should not have been.")
        self.assertGreater(len(wav2vec_files), 0, "Wav2Vec2.0 files were not created as requested.")

        # Verify that the metadata correctly reflects the requested features
        meta_path = os.path.join(out_dir, 'meta.json')
        with open(meta_path, 'r') as f:
            meta = json.load(f)
        self.assertEqual(meta['processing_parameters']['features_extracted'], ['wav2vec'])

if __name__ == '__main__':
    unittest.main() 