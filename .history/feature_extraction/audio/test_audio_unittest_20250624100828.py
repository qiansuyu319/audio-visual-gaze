import unittest
import os
import numpy as np
import tempfile
import shutil
import subprocess
import json

# Define the absolute path to the directory containing the audio scripts
AUDIO_DIR = os.path.dirname(__file__)
# Define the absolute path to the main processing script to be tested
PROCESS_SCRIPT = os.path.join(AUDIO_DIR, 'process_audio.py')
# Define the path to the real audio file, assuming the test is run from the project root
REAL_WAV_PATH = os.path.join(os.path.dirname(AUDIO_DIR), '..', 'data', '012.wav')

class TestRealAudioProcessing(unittest.TestCase):
    """
    A test suite for the unified `process_audio.py` script using a real audio file.
    """
    
    @classmethod
    def setUpClass(cls):
        """
        Set up the test environment by creating a temporary directory for output
        and running the processing script once for all tests in this class.
        """
        cls.tmpdir = tempfile.mkdtemp()
        cls.out_dir = os.path.join(cls.tmpdir, 'real_audio_output')
        
        # Verify that the test audio file exists before running the test
        if not os.path.exists(REAL_WAV_PATH):
            raise FileNotFoundError(f"Test audio file not found at: {REAL_WAV_PATH}")

        # Execute the processing script
        cmd = ['python', PROCESS_SCRIPT, '--input_audio', REAL_WAV_PATH, '--output_dir', cls.out_dir, '--cpu']
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print(result.stderr)

    @classmethod
    def tearDownClass(cls):
        """Cleans up and removes the temporary directory after all tests are complete."""
        shutil.rmtree(cls.tmpdir)

    def test_meta_file_creation_and_content(self):
        """
        Validates that 'meta.json' is created and contains correct, plausible data.
        """
        meta_path = os.path.join(self.out_dir, 'meta.json')
        self.assertTrue(os.path.exists(meta_path), "meta.json was not created.")
        
        with open(meta_path, 'r') as f:
            meta = json.load(f)
        
        # Check metadata content
        self.assertIn('source_audio', meta)
        self.assertTrue(meta['source_audio'].endswith('012.wav'))
        params = meta['processing_parameters']
        self.assertGreater(params['total_windows'], 0, "Metadata reports zero windows processed.")
        self.assertEqual(params['target_sr'], 16000)

    def test_feature_file_creation(self):
        """
        Validates that the correct number and type of feature files were generated.
        """
        with open(os.path.join(self.out_dir, 'meta.json'), 'r') as f:
            num_windows = json.load(f)['processing_parameters']['total_windows']

        egemaps_files = [f for f in os.listdir(self.out_dir) if f.endswith('_egemaps.npy')]
        wav2vec_files = [f for f in os.listdir(self.out_dir) if f.endswith('_wav2vec.npy')]
        
        self.assertEqual(len(egemaps_files), num_windows, "Number of eGeMAPS files does not match window count.")
        self.assertEqual(len(wav2vec_files), num_windows, "Number of Wav2Vec2.0 files does not match window count.")

    def test_feature_file_content(self):
        """
        Validates the content and dimensionality of the first feature file of each type.
        """
        egemaps_file = os.path.join(self.out_dir, 'window_00000_egemaps.npy')
        wav2vec_file = os.path.join(self.out_dir, 'window_00000_wav2vec.npy')
        
        self.assertTrue(os.path.exists(egemaps_file), "First eGeMAPS feature file is missing.")
        self.assertTrue(os.path.exists(wav2vec_file), "First Wav2Vec2.0 feature file is missing.")
        
        # Check feature shapes
        egemaps_feat = np.load(egemaps_file)
        wav2vec_feat = np.load(wav2vec_file)
        self.assertEqual(egemaps_feat.shape, (88,), "eGeMAPS feature shape is incorrect.")
        self.assertEqual(wav2vec_feat.shape, (768,), "Wav2Vec2.0 feature shape is incorrect.")
        
        # Check for NaN or Inf values
        self.assertFalse(np.isnan(egemaps_feat).any(), "eGeMAPS features contain NaN values.")
        self.assertFalse(np.isinf(egemaps_feat).any(), "eGeMAPS features contain Inf values.")
        self.assertFalse(np.isnan(wav2vec_feat).any(), "Wav2Vec2.0 features contain NaN values.")
        self.assertFalse(np.isinf(wav2vec_feat).any(), "Wav2Vec2.0 features contain Inf values.")

if __name__ == '__main__':
    unittest.main() 