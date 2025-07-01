# Placeholder for CLIP feature extraction

import argparse
import numpy as np
import os
from PIL import Image
from tqdm import tqdm
import torch

from transformers import AutoProcessor, AutoModel

def load_mclip_vision_model(model_name="M-CLIP/XLM-Roberta-Large-Vit-B-16Plus", device="cuda"):
    try:
        processor = AutoProcessor.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name).vision_model
        model.eval().to(device)
        def preprocess_fn(img):
            return processor(images=img, return_tensors="pt")['pixel_values'][0]
        return model, preprocess_fn
    except Exception as e:
        raise RuntimeError(f"Failed to load M-CLIP model from Huggingface: {e}\nPlease ensure the model name is correct and transformers is up to date.")

def preprocess_image(image, preprocess_fn):
    if isinstance(image, str):
        image = Image.open(image).convert('RGB')
    return preprocess_fn(image)

def extract_clip_features(input_dir, output_npy, model_name="M-CLIP/XLM-Roberta-Large-Vit-B-16Plus", batch_size=32, device=None):
    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
    if not files:
        raise ValueError("No image files found in the input directory.")
    try:
        from natsort import natsorted
        files = natsorted(files)
    except ImportError:
        files.sort()
    print(f"Loaded {len(files)} images from {input_dir}")
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, preprocess_fn = load_mclip_vision_model(model_name, device)
    all_features = []
    for i in tqdm(range(0, len(files), batch_size), desc="Extracting visual features"):
        batch_files = files[i:i+batch_size]
        batch_imgs = []
        for fname in batch_files:
            fpath = os.path.join(input_dir, fname)
            try:
                img = preprocess_image(fpath, preprocess_fn)
                batch_imgs.append(img)
            except Exception as e:
                print(f"[WARN] Failed to process {fname}: {e}")
        if not batch_imgs:
            continue
        imgs_tensor = torch.stack(batch_imgs).to(device)
        with torch.no_grad():
            if hasattr(model, 'forward_features'):
                feats = model.forward_features(imgs_tensor)
                if isinstance(feats, dict):
                    feats = feats['x'] if 'x' in feats else list(feats.values())[0]
            else:
                feats = model(imgs_tensor)
            if isinstance(feats, (tuple, list)):
                feats = feats[0]
        all_features.append(feats.cpu().numpy())
    if not all_features:
        raise RuntimeError("No features extracted from images.")
    features = np.concatenate(all_features, axis=0)
    np.save(output_npy, features)
    print(f"Saved features for {features.shape[0]} images to {output_npy}, shape: {features.shape}")

def main():
    parser = argparse.ArgumentParser(description="Extract M-CLIP visual features from a folder of images (Huggingface only).")
    parser.add_argument('--input_dir', type=str, required=True, help='Input directory of images (jpg/png).')
    parser.add_argument('--output_npy', type=str, required=True, help='Output .npy file for features.')
    parser.add_argument('--model', type=str, default='M-CLIP/XLM-Roberta-Large-Vit-B-16Plus', help='M-CLIP model name (Huggingface).')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for feature extraction.')
    parser.add_argument('--device', type=str, default=None, help='Device for computation (cuda/cpu).')
    args = parser.parse_args()
    extract_clip_features(
        input_dir=args.input_dir,
        output_npy=args.output_npy,
        model_name=args.model,
        batch_size=args.batch_size,
        device=args.device
    )

if __name__ == "__main__":
    main() 