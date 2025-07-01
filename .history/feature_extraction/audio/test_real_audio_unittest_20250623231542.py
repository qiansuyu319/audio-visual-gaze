import unittest
import os
import numpy as np
import shutil
import librosa

class TestRealAudioFeatureExtraction(unittest.TestCase):
    def test_real_audio_egemaps(self):
        audio_path = os.path.abspath('data/012.wav')
        output_dir = 'test_real_audio_egemaps'
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        os.makedirs(output_dir, exist_ok=True)
        # eGeMAPS特征提取
        ret = os.system(f'python feature_extraction/audio/extract_egemaps.py --input_wav "{audio_path}" --output_dir "{output_dir}"')
        self.assertEqual(ret, 0, "extract_egemaps.py did not run successfully")
        files = [f for f in os.listdir(output_dir) if f.endswith('.npy')]
        self.assertGreater(len(files), 0, "No egemaps features extracted from real audio")
        arr = np.load(os.path.join(output_dir, files[0]))
        self.assertEqual(arr.shape, (1, 88))
        self.assertFalse(np.isnan(arr).any())
        self.assertFalse(np.isinf(arr).any())
        self.assertGreater(arr.std(), 0)
        self.egemaps_files = files
        self.egemaps_dir = output_dir
        self.egemaps_count = len(files)

    def test_real_audio_wav2vec(self):
        audio_path = os.path.abspath('data/012.wav')
        output_dir = 'test_real_audio_wav2vec'
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        os.makedirs(output_dir, exist_ok=True)
        # wav2vec2.0特征提取
        ret = os.system(f'python feature_extraction/audio/extract_wav2vec.py --input_wav "{audio_path}" --output_dir "{output_dir}"')
        self.assertEqual(ret, 0, "extract_wav2vec.py did not run successfully")
        files = [f for f in os.listdir(output_dir) if f.endswith('.npy')]
        self.assertGreater(len(files), 0, "No wav2vec features extracted from real audio")
        arr = np.load(os.path.join(output_dir, files[0]))
        self.assertIn(arr.ndim, [1, 2])
        self.assertEqual(arr.shape[-1], 768)
        self.assertFalse(np.isnan(arr).any())
        self.assertFalse(np.isinf(arr).any())
        self.assertGreater(arr.std(), 0)
        self.wav2vec_files = files
        self.wav2vec_dir = output_dir
        self.wav2vec_count = len(files)

    def test_real_audio_preprocess(self):
        audio_path = os.path.abspath('data/012.wav')
        output_dir = 'test_real_audio_chunks'
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        os.makedirs(output_dir, exist_ok=True)
        # 音频分割归一化
        ret = os.system(f'python feature_extraction/audio/preprocess_audio.py --input_wav "{audio_path}" --output_dir "{output_dir}" --chunk_length 1.0')
        self.assertEqual(ret, 0, "preprocess_audio.py did not run successfully")
        files = [f for f in os.listdir(output_dir) if f.endswith('.wav')]
        self.assertGreater(len(files), 0, "No audio chunks generated from real audio")
        self.chunks_files = files
        self.chunks_dir = output_dir
        self.chunks_count = len(files)

    def test_alignment(self):
        # 检查三个功能的窗口/分块数量是否一致
        egemaps_dir = 'test_real_audio_egemaps'
        wav2vec_dir = 'test_real_audio_wav2vec'
        chunks_dir = 'test_real_audio_chunks'
        egemaps_files = [f for f in os.listdir(egemaps_dir) if f.endswith('.npy')]
        wav2vec_files = [f for f in os.listdir(wav2vec_dir) if f.endswith('.npy')]
        chunks_files = [f for f in os.listdir(chunks_dir) if f.endswith('.wav')]
        self.assertEqual(len(egemaps_files), len(wav2vec_files), f"Window count mismatch: eGeMAPS={len(egemaps_files)}, wav2vec2.0={len(wav2vec_files)}")
        self.assertEqual(len(egemaps_files), len(chunks_files), f"Window count mismatch: eGeMAPS={len(egemaps_files)}, chunks={len(chunks_files)}")

if __name__ == '__main__':
    unittest.main() 