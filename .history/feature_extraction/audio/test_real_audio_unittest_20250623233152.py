import unittest
import os
import numpy as np

class TestAudioFeatureWindowAlignment(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.egemaps_dir = 'test_real_audio_egemaps'
        cls.wav2vec_dir = 'test_real_audio_wav2vec'
        assert os.path.exists(cls.egemaps_dir), f"eGeMAPS dir not found: {cls.egemaps_dir}"
        assert os.path.exists(cls.wav2vec_dir), f"wav2vec dir not found: {cls.wav2vec_dir}"

    def test_window_count_and_names(self):
        egemaps_files = sorted([f for f in os.listdir(self.egemaps_dir) if f.endswith('.npy')])
        wav2vec_files = sorted([f for f in os.listdir(self.wav2vec_dir) if f.endswith('.npy')])
        self.assertGreater(len(egemaps_files), 0, "No eGeMAPS features found")
        self.assertGreater(len(wav2vec_files), 0, "No wav2vec2.0 features found")
        self.assertEqual(len(egemaps_files), len(wav2vec_files), 
                         f"Feature count mismatch: eGeMAPS={len(egemaps_files)}, wav2vec2.0={len(wav2vec_files)}")
        # 检查文件编号一一对应
        for e_file, w_file in zip(egemaps_files, wav2vec_files):
            self.assertEqual(e_file, w_file, f"Window index mismatch: {e_file} vs {w_file}")

    def test_feature_shapes(self):
        egemaps_files = sorted([f for f in os.listdir(self.egemaps_dir) if f.endswith('.npy')])
        wav2vec_files = sorted([f for f in os.listdir(self.wav2vec_dir) if f.endswith('.npy')])
        for e_file, w_file in zip(egemaps_files, wav2vec_files):
            e_path = os.path.join(self.egemaps_dir, e_file)
            w_path = os.path.join(self.wav2vec_dir, w_file)
            e_feat = np.load(e_path)
            w_feat = np.load(w_path)
            self.assertEqual(e_feat.shape, (1, 88), f"eGeMAPS shape error: {e_file} got {e_feat.shape}")
            # wav2vec shape可以为(1, 768)或(768,)
            self.assertIn(w_feat.shape, [(1, 768), (768,)], f"wav2vec2.0 shape error: {w_file} got {w_feat.shape}")

    def test_feature_quality(self):
        egemaps_files = sorted([f for f in os.listdir(self.egemaps_dir) if f.endswith('.npy')])
        wav2vec_files = sorted([f for f in os.listdir(self.wav2vec_dir) if f.endswith('.npy')])
        for e_file, w_file in zip(egemaps_files, wav2vec_files):
            e_feat = np.load(os.path.join(self.egemaps_dir, e_file))
            w_feat = np.load(os.path.join(self.wav2vec_dir, w_file))
            # nan/inf 检查
            self.assertFalse(np.isnan(e_feat).any(), f"eGeMAPS nan in {e_file}")
            self.assertFalse(np.isnan(w_feat).any(), f"wav2vec2.0 nan in {w_file}")
            self.assertFalse(np.isinf(e_feat).any(), f"eGeMAPS inf in {e_file}")
            self.assertFalse(np.isinf(w_feat).any(), f"wav2vec2.0 inf in {w_file}")
            # 非常量向量检查
            self.assertGreater(e_feat.std(), 0, f"eGeMAPS std=0 in {e_file}")
            self.assertGreater(w_feat.std(), 0, f"wav2vec2.0 std=0 in {w_file}")

if __name__ == '__main__':
    unittest.main()
