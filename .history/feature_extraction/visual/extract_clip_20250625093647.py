# Placeholder for CLIP feature extraction

import os
import argparse
from PIL import Image
import torch
import numpy as np
from tqdm import tqdm

try:
    from transformers import AutoProcessor, AutoModel
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    from torchvision import transforms
    import timm

def load_mclip_vision_model(model_name="M-CLIP/XLM-Roberta-Large-Vit-B-16Plus", device="cuda"):
    if HF_AVAILABLE:
        processor = AutoProcessor.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name).vision_model
        model.eval().to(device)
        def preprocess_fn(img):
            return processor(images=img, return_tensors="pt")['pixel_values'][0]
        return model, preprocess_fn
    else:
        # Fallback: use timm ViT and torchvision transforms
        model = timm.create_model('vit_base_patch16_224', pretrained=True)
        model.eval().to(device)
        preprocess_fn = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        return model, preprocess_fn

def preprocess_image(image, preprocess_fn):
    if isinstance(image, str):
        image = Image.open(image).convert('RGB')
    return preprocess_fn(image)

def extract_visual_feature(image, vision_model, preprocess_fn, device="cuda"):
    img_tensor = preprocess_image(image, preprocess_fn).to(device)
    with torch.no_grad():
        if hasattr(vision_model, 'forward_features'):
            features = vision_model.forward_features(img_tensor.unsqueeze(0))
            if isinstance(features, dict):
                features = features['x'] if 'x' in features else list(features.values())[0]
        else:
            features = vision_model(img_tensor.unsqueeze(0))
        if isinstance(features, (tuple, list)):
            features = features[0]
    return features.squeeze(0).cpu().numpy()

def extract_from_folder(frame_dir, save_path, vision_model, preprocess_fn, device="cuda", batch_size=16):
    files = [f for f in os.listdir(frame_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
    try:
        from natsort import natsorted
        files = natsorted(files)
    except ImportError:
        files.sort()
    features = []
    failed = []
    for fname in tqdm(files, desc="Extracting features"):
        fpath = os.path.join(frame_dir, fname)
        try:
            feat = extract_visual_feature(fpath, vision_model, preprocess_fn, device)
            features.append(feat)
        except Exception as e:
            print(f"[WARN] Failed to process {fname}: {e}")
            failed.append(fname)
    features = np.stack(features)
    np.save(save_path, features)
    print(f"Saved features for {len(features)} images to {save_path}")
    if failed:
        print(f"Failed to process {len(failed)} images. See log above.")

def main():
    parser = argparse.ArgumentParser(description="Extract visual features using M-CLIP vision model.")
    parser.add_argument('--input', type=str, required=True, help='Input image file or folder of images.')
    parser.add_argument('--output', type=str, required=True, help='Output .npy file to save features.')
    parser.add_argument('--model', type=str, default="M-CLIP/XLM-Roberta-Large-Vit-B-16Plus", help='M-CLIP model name.')
    parser.add_argument('--device', type=str, default=None, help='Device for computation (cuda/cpu).')
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    vision_model, preprocess_fn = load_mclip_vision_model(args.model, device)

    if os.path.isdir(args.input):
        extract_from_folder(args.input, args.output, vision_model, preprocess_fn, device)
    elif os.path.isfile(args.input):
        feat = extract_visual_feature(args.input, vision_model, preprocess_fn, device)
        np.save(args.output, feat)
        print(f"Saved feature for {args.input} to {args.output}")
    else:
        print(f"Input path {args.input} not found.")

if __name__ == '__main__':
    main() 