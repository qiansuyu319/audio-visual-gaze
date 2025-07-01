# M-CLIP Text Embedding Extraction Script
# This script extracts multilingual CLIP (M-CLIP) text embeddings for a list of sentences.
# It provides a command-line interface for specifying input and output files.
# Embeddings are saved as a .npy file, and optionally a .json mapping file.

import argparse
import numpy as np
import json
from sentence_transformers import SentenceTransformer


def extract_mclip_embeddings(input_txt, output_npy, output_json=None, model_name="M-CLIP/XLM-Roberta-Large-Vit-B-16Plus"):
    """
    Extracts M-CLIP text embeddings for each line in the input text file.

    Args:
        input_txt (str): Path to the input text file (one sentence per line).
        output_npy (str): Path to save the numpy array of embeddings.
        output_json (str, optional): Path to save the mapping of lines to indices.
        model_name (str): Name of the M-CLIP model to use.
    """
    # Load sentences
    with open(input_txt, 'r', encoding='utf-8') as f:
        sentences = [line.strip() for line in f if line.strip()]

    # Load M-CLIP model
    print(f"Loading M-CLIP model: {model_name}")
    model = SentenceTransformer(model_name)

    # Compute embeddings
    print(f"Encoding {len(sentences)} sentences...")
    embeddings = model.encode(sentences, show_progress_bar=True, convert_to_numpy=True)

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
    parser = argparse.ArgumentParser(description="Extract M-CLIP text embeddings from a text file.")
    parser.add_argument('--input_txt', type=str, required=True, help='Input text file (one sentence per line).')
    parser.add_argument('--output_npy', type=str, required=True, help='Output .npy file for embeddings.')
    parser.add_argument('--output_json', type=str, default=None, help='Optional output .json for line mapping.')
    parser.add_argument('--model', type=str, default="M-CLIP/XLM-Roberta-Large-Vit-B-16Plus", help='M-CLIP model name.')
    args = parser.parse_args()

    extract_mclip_embeddings(
        input_txt=args.input_txt,
        output_npy=args.output_npy,
        output_json=args.output_json,
        model_name=args.model
    )

if __name__ == '__main__':
    main() 