import argparse
import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
import open_clip

def load_clip_model(model_name='ViT-B-16-plus-240', pretrained='laion400m_e32', device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained, device=device)
    model.eval()
    return model, preprocess, device

def extract_visual_features_from_folder(frame_dir, save_path, model, preprocess, device, batch_size=16):
    img_files = sorted([
        f for f in os.listdir(frame_dir)
        if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))
    ])
    if not img_files:
        raise ValueError(f"No image files found in {frame_dir}")
    features = []
    for i in tqdm(range(0, len(img_files), batch_size), desc="Extracting visual features"):
        batch_files = img_files[i:i+batch_size]
        batch_imgs = []
        for fname in batch_files:
            img_path = os.path.join(frame_dir, fname)
            try:
                img = Image.open(img_path).convert('RGB')
                batch_imgs.append(preprocess(img))
            except Exception as e:
                print(f"[WARN] Failed to open {fname}: {e}")
        if not batch_imgs:
            continue
            
        batch_tensor = torch.stack(batch_imgs).to(device)

        with torch.no_grad():
            feats = model.encode_image(batch_tensor)  # (B, D)
        features.append(feats.cpu().numpy())
    if not features:
        raise RuntimeError("No features extracted from images.")
    arr = np.concatenate(features, axis=0)
    np.save(save_path, arr)
    print(f"Saved {len(arr)} visual features to {save_path}")

def main():
    parser = argparse.ArgumentParser(description="Extract visual features from a folder of images using open_clip.")
    parser.add_argument('--input_dir', type=str, required=True, help='Input directory of images (jpg/png).')
    parser.add_argument('--output_npy', type=str, required=True, help='Output .npy file for features. 推荐放在 output/visual_feature/')
    parser.add_argument('--model_name', type=str, default='ViT-B-16-plus-240', help='Model name from open_clip.')
    parser.add_argument('--pretrained', type=str, default='laion400m_e32', help='Pretrained weights from open_clip.')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for feature extraction.')
    parser.add_argument('--device', type=str, default=None, help='Device for computation (cuda/cpu).')
    args = parser.parse_args()
    model, preprocess, device = load_clip_model(args.model_name, args.pretrained, args.device)
    extract_visual_features_from_folder(args.input_dir, args.output_npy, model, preprocess, device, args.batch_size)

if __name__ == "__main__":
    main()
