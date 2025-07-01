# Placeholder for CLIP feature extraction

import argparse
import numpy as np
import os
from PIL import Image
from tqdm import tqdm
import torch
from multilingual_clip import pt_multilingual_clip
import transformers

def preprocess_image(image, image_size=224):
    if isinstance(image, str):
        image = Image.open(image).convert('RGB')
    image = image.resize((image_size, image_size))
    img = np.array(image).astype(np.float32) / 255.0
    # Normalize as in ViT
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = (img - mean) / std
    img = img.transpose(2, 0, 1)  # HWC to CHW
    return torch.tensor(img, dtype=torch.float32)

def extract_clip_visual_features(input_dir, output_npy, model_name='M-CLIP/XLM-Roberta-Large-Vit-B-16Plus', batch_size=32, device=None):
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
    model = pt_multilingual_clip.MultilingualCLIP.from_pretrained(model_name)
    model = model.to(device)
    model.eval()
    all_features = []
    image_size = 224  # M-CLIP ViT default
    for i in tqdm(range(0, len(files), batch_size), desc="Extracting visual features"):
        batch_files = files[i:i+batch_size]
        batch_imgs = []
        for fname in batch_files:
            fpath = os.path.join(input_dir, fname)
            try:
                img = preprocess_image(fpath, image_size)
                batch_imgs.append(img)
            except Exception as e:
                print(f"[WARN] Failed to process {fname}: {e}")
        if not batch_imgs:
            continue
        imgs_tensor = torch.stack(batch_imgs).to(device)
        with torch.no_grad():
            feats = model.vision_model(imgs_tensor)
            if isinstance(feats, (tuple, list)):
                feats = feats[0]
        all_features.append(feats.cpu().numpy())
    if not all_features:
        raise RuntimeError("No features extracted from images.")
    features = np.concatenate(all_features, axis=0)
    np.save(output_npy, features)
    print(f"Saved visual features for {features.shape[0]} images to {output_npy}, shape: {features.shape}")

def extract_clip_text_features(input_text, output_npy, model_name='M-CLIP/XLM-Roberta-Large-Vit-B-16Plus', batch_size=32, device=None):
    if not os.path.exists(input_text):
        raise FileNotFoundError(f"Input text file not found: {input_text}")
    with open(input_text, 'r', encoding='utf-8') as f:
        texts = [line.strip() for line in f if line.strip()]
    if not texts:
        raise ValueError("No valid lines found in the input text file.")
    print(f"Loaded {len(texts)} lines from {input_text}")
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = pt_multilingual_clip.MultilingualCLIP.from_pretrained(model_name)
    model = model.to(device)
    model.eval()
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    all_embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Extracting text features"):
        batch = texts[i:i+batch_size]
        with torch.no_grad():
            emb = model.forward(batch, tokenizer)
            all_embeddings.append(emb.detach().cpu().numpy())
    embeddings = np.concatenate(all_embeddings, axis=0)
    np.save(output_npy, embeddings)
    print(f"Saved text features for {len(texts)} texts to {output_npy}, shape: {embeddings.shape}")

def main():
    parser = argparse.ArgumentParser(description="Extract M-CLIP visual or text features using M-CLIP official model.")
    parser.add_argument('--input_dir', type=str, help='Input directory of images (jpg/png).')
    parser.add_argument('--input_text', type=str, help='Input text file (one sentence per line).')
    parser.add_argument('--output_npy', type=str, required=True, help='Output .npy file for features.')
    parser.add_argument('--model', type=str, default='M-CLIP/XLM-Roberta-Large-Vit-B-16Plus', help='M-CLIP model name.')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for feature extraction.')
    parser.add_argument('--device', type=str, default=None, help='Device for computation (cuda/cpu).')
    args = parser.parse_args()
    if args.input_dir:
        extract_clip_visual_features(
            input_dir=args.input_dir,
            output_npy=args.output_npy,
            model_name=args.model,
            batch_size=args.batch_size,
            device=args.device
        )
    if args.input_text:
        extract_clip_text_features(
            input_text=args.input_text,
            output_npy=args.output_npy,
            model_name=args.model,
            batch_size=args.batch_size,
            device=args.device
        )
    if not args.input_dir and not args.input_text:
        print("Please provide either --input_dir for images or --input_text for text.")

if __name__ == "__main__":
    main() 