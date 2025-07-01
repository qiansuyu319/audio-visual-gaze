import argparse
import numpy as np
from multilingual_clip import pt_multilingual_clip
from transformers import AutoTokenizer
import torch
import os

def extract_mclip_embeddings(input_txt, output_npy, model_name='M-CLIP/XLM-Roberta-Large-Vit-B-16Plus', batch_size=32):
    """
    Extract M-CLIP embeddings for each line in the input text file and save as .npy.
    Args:
        input_txt (str): Path to the input text file (one sentence per line).
        output_npy (str): Path to save the numpy array of embeddings.
        model_name (str): Name of the M-CLIP model to use.
        batch_size (int): Number of sentences per batch for embedding extraction.
    """
    if not os.path.exists(input_txt):
        raise FileNotFoundError(f"Input text file not found: {input_txt}")
    with open(input_txt, 'r', encoding='utf-8') as f:
        texts = [line.strip() for line in f if line.strip()]
    if not texts:
        raise ValueError("No valid lines found in the input text file.")
    print(f"Loaded {len(texts)} lines from {input_txt}")
    model = pt_multilingual_clip.MultilingualCLIP.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        with torch.no_grad():
            emb = model.forward(batch, tokenizer)
            all_embeddings.append(emb.detach().cpu().numpy())
    embeddings = np.concatenate(all_embeddings, axis=0)
    np.save(output_npy, embeddings)
    print(f"Saved embeddings for {len(texts)} texts to {output_npy}, shape: {embeddings.shape}")

def main():
    parser = argparse.ArgumentParser(description="Extract M-CLIP embeddings from a text file using PyTorch backend.")
    parser.add_argument('--input_txt', type=str, required=True, help='Input text file (one sentence per line).')
    parser.add_argument('--output_npy', type=str, required=True, help='Output .npy file for embeddings.')
    parser.add_argument('--model', type=str, default='M-CLIP/XLM-Roberta-Large-Vit-B-16Plus', help='M-CLIP model name.')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for embedding extraction.')
    args = parser.parse_args()
    extract_mclip_embeddings(
        input_txt=args.input_txt,
        output_npy=args.output_npy,
        model_name=args.model,
        batch_size=args.batch_size
    )

if __name__ == "__main__":
    main()
