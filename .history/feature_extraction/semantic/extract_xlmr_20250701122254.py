# XLM-R Text Embedding Extraction Script
# This script extracts XLM-RoBERTa text embeddings for a list of sentences.
# It provides a command-line interface for specifying input and output files.
# Embeddings are saved as a .npy file, and optionally a .json mapping file.

# 推荐输出目录：output/semantic_feature/
# 建议：--output_npy output/semantic_feature/xlmr_embeddings.npy

import argparse
import numpy as np
import json
import torch
from transformers import AutoTokenizer, AutoModel


def extract_xlmr_embeddings(input_txt, output_npy, output_json=None, model_name="xlm-roberta-base", device=None):
    """
    Extracts XLM-R text embeddings for each line in the input text file.

    Args:
        input_txt (str): Path to the input text file (one sentence per line).
        output_npy (str): Path to save the numpy array of embeddings.
        output_json (str, optional): Path to save the mapping of lines to indices.
        model_name (str): Name of the XLM-R model to use.
        device (str, optional): Device to run the model on ("cuda" or "cpu").
    """
    # Load sentences
    with open(input_txt, 'r', encoding='utf-8') as f:
        sentences = [line.strip() for line in f if line.strip()]

    # Set device
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load XLM-R model and tokenizer
    print(f"Loading XLM-R model: {model_name} on {device}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()

    # Compute embeddings
    print(f"Encoding {len(sentences)} sentences...")
    embeddings = []
    with torch.no_grad():
        for sent in sentences:
            inputs = tokenizer(sent, return_tensors="pt", truncation=True, padding=True).to(device)
            outputs = model(**inputs)
            # Use the [CLS] token representation as the sentence embedding
            emb = outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy()
            embeddings.append(emb)
    embeddings = np.stack(embeddings)

    # Save embeddings
    np.save(output_npy, embeddings)
    print(f"Embeddings saved to: {output_npy}")

    # Optionally save mapping
    if output_json:
        mapping = {i: sent for i, sent in enumerate(sentences)}
        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump(mapping, f, ensure_ascii=False, indent=2)
        print(f"Mapping saved to: {output_json}")


def main():
    parser = argparse.ArgumentParser(description="Extract XLM-R text embeddings from a text file.")
    parser.add_argument('--input_txt', type=str, required=True, help='Input text file (one sentence per line).')
    parser.add_argument('--output_npy', type=str, required=True, help='Output .npy file for embeddings. 推荐放在 output/semantic_feature/')
    parser.add_argument('--output_json', type=str, default=None, help='Optional output .json for line mapping.')
    parser.add_argument('--model', type=str, default="xlm-roberta-base", help='XLM-R model name.')
    parser.add_argument('--device', type=str, default=None, help='Device for computation ("cuda" or "cpu").')
    args = parser.parse_args()

    extract_xlmr_embeddings(
        input_txt=args.input_txt,
        output_npy=args.output_npy,
        output_json=args.output_json,
        model_name=args.model,
        device=args.device
    )

if __name__ == '__main__':
    main() 