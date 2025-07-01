import os
from PIL import Image
import torch
import numpy as np
from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm

# Load M-CLIP vision backbone and processor
model_id = "M-CLIP/XLM-Roberta-Large-Vit-L-14"
model = CLIPModel.from_pretrained(model_id)
processor = CLIPProcessor.from_pretrained(model_id)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

def extract_visual_features_from_folder(frame_dir, save_path):
    img_files = sorted([
        f for f in os.listdir(frame_dir)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ])
    features = []
    for fname in tqdm(img_files, desc="Extracting visual features"):
        img_path = os.path.join(frame_dir, fname)
        img = Image.open(img_path).convert('RGB')
        inputs = processor(images=img, return_tensors="pt").to(device)
        with torch.no_grad():
            feat = model.get_image_features(**inputs)  # (1, D)
        features.append(feat.cpu().numpy().squeeze(0))  # (D,)
    arr = np.stack(features)
    np.save(save_path, arr)
    print(f"Saved {len(arr)} visual features to {save_path}")

# Example usage:
# extract_visual_features_from_folder("/path/to/frames", "visual_features.npy")
