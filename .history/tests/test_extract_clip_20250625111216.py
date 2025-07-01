import os
import unittest
import numpy as np
from PIL import Image
from feature_extraction.visual.extract_clip import load_clip_model, extract_visual_features_from_folder

class TestExtractClip(unittest.TestCase):
    def setUp(self):
        self.input_dir = 'test_images'
        self.output_npy = 'test_features.npy'
        os.makedirs(self.input_dir, exist_ok=True)
        # Create a dummy image
        self.image_path = os.path.join(self.input_dir, 'test_image.png')
        Image.new('RGB', (100, 100), color = 'red').save(self.image_path)

    def tearDown(self):
        os.remove(self.image_path)
        os.rmdir(self.input_dir)
        if os.path.exists(self.output_npy):
            os.remove(self.output_npy)

    def test_extract_visual_features(self):
        model, preprocess, device = load_clip_model()
        extract_visual_features_from_folder(self.input_dir, self.output_npy, model, preprocess, device)
        self.assertTrue(os.path.exists(self.output_npy))
        features = np.load(self.output_npy)
        self.assertEqual(features.shape, (1, 640))
        self.assertEqual(features.shape, (1, 768))

if __name__ == '__main__':
    unittest.main() 