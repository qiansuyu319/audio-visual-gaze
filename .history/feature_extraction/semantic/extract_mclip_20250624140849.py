# M-CLIP Text Embedding Extraction Script (TensorFlow & PyTorch)
# This script extracts multilingual CLIP (M-CLIP) text embeddings for a list of sentences.
# It supports both TensorFlow and PyTorch backends, selectable via CLI.
# Embeddings are saved as a .npy file, and optionally a .json mapping file.

import argparse
import numpy as np
import json
import transformers


def extract_mclip_embeddings_pt(sentences, model_name):
    """
    Extract M-CLIP embeddings using PyTorch backend.
    Args:
        sentences (list): List of input sentences.
        model_name (str): Name of the M-CLIP model.
    Returns:
        np.ndarray: Embedding matrix.
    """
    from multilingual_clip import pt_multilingual_clip
    import torch
    model = pt_multilingual_clip.MultilingualCLIP.from_pretrained(model_name)
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    with torch.no_grad():
        embeddings = model.forward(sentences, tokenizer)
        return embeddings.cpu().numpy()


def extract_mclip_embeddings_tf(sentences, model_name):
    """
    Extract M-CLIP embeddings using TensorFlow backend.
    Args:
        sentences (list): List of input sentences.
        model_name (str): Name of the M-CLIP model.
    Returns:
        np.ndarray: Embedding matrix.
    """
    from multilingual_clip import tf_multilingual_clip
    import tensorflow as tf
    model = tf_multilingual_clip.MultiLingualCLIP.from_pretrained(model_name)
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    inData = tokenizer.batch_encode_plus(sentences, return_tensors='tf', padding=True)
    embeddings = model(inData)
    return embeddings.numpy()


def extract_mclip_embeddings(input_txt, output_npy, output_json=None, model_name="M-CLIP/XLM-Roberta-Large-Vit-B-16Plus", backend="pt"):
    """
    Extracts M-CLIP text embeddings for each line in the input text file using the selected backend.
    Args:
        input_txt (str): Path to the input text file (one sentence per line).
        output_npy (str): Path to save the numpy array of embeddings.
        output_json (str, optional): Path to save the mapping of lines to indices.
        model_name (str): Name of the M-CLIP model to use.
        backend (str): 'pt' for PyTorch, 'tf' for TensorFlow.
    """
    with open(input_txt, 'r', encoding='utf-8') as f:
        sentences = [line.strip() for line in f if line.strip()]
    print(f"Extracting M-CLIP embeddings using backend: {backend}")
    if backend == "pt":
        embeddings = extract_mclip_embeddings_pt(sentences, model_name)
    elif backend == "tf":
        embeddings = extract_mclip_embeddings_tf(sentences, model_name)
    else:
        raise ValueError("Unsupported backend. Use 'pt' for PyTorch or 'tf' for TensorFlow.")
    np.save(output_npy, embeddings)
    print(f"Embeddings saved to: {output_npy}")
    if output_json:
        mapping = {i: sent for i, sent in enumerate(sentences)}
        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump(mapping, f, ensure_ascii=False, indent=2)
        print(f"Mapping saved to: {output_json}")


def main():
    parser = argparse.ArgumentParser(description="Extract M-CLIP text embeddings from a text file using multilingual-clip (PyTorch or TensorFlow backend).")
    parser.add_argument('--input_txt', type=str, required=True, help='Input text file (one sentence per line).')
    parser.add_argument('--output_npy', type=str, required=True, help='Output .npy file for embeddings.')
    parser.add_argument('--output_json', type=str, default=None, help='Optional output .json for line mapping.')
    parser.add_argument('--model', type=str, default="M-CLIP/XLM-Roberta-Large-Vit-B-16Plus", help='M-CLIP model name.')
    parser.add_argument('--backend', type=str, default="pt", choices=["pt", "tf"], help="Backend: 'pt' for PyTorch, 'tf' for TensorFlow.")
    args = parser.parse_args()

    extract_mclip_embeddings(
        input_txt=args.input_txt,
        output_npy=args.output_npy,
        output_json=args.output_json,
        model_name=args.model,
        backend=args.backend
    )

if __name__ == '__main__':
    main() 