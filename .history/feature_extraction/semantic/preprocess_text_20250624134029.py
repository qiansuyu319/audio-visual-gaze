# Semantic Feature Extraction Pipeline Script
# This script provides a full pipeline for semantic feature extraction from audio:
# 1. Transcribe audio to text using WhisperX
# 2. Preprocess the transcript (cleaning, segmentation)
# 3. Extract M-CLIP and XLM-R embeddings from the preprocessed text
# The script provides a command-line interface for configuring each stage and input/output paths.

import argparse
import os
import sys
import subprocess

# Import the other scripts as modules (assume they are in the same directory)
from extract_whisperx import transcribe_with_whisperx
from extract_mclip import extract_mclip_embeddings
from extract_xlmr import extract_xlmr_embeddings
from multilingual_clip.multilingual_clip import MultilingualCLIP

def pipeline(
    input_audio,
    output_dir,
    whisperx_model="large-v2",
    language=None,
    batch_size=16,
    device=None,
    compute_type=None,
    preprocess_lowercase=True,
    preprocess_remove_punct=True,
    preprocess_segment=False,
    mclip_model="M-CLIP/XLM-Roberta-Large-Vit-B-16Plus",
    xlmr_model="xlm-roberta-base"
):
    """
    Full pipeline: audio -> transcript -> preprocessed text -> M-CLIP & XLM-R embeddings.
    Args:
        input_audio (str): Path to input audio file.
        output_dir (str): Directory to save all outputs.
        whisperx_model (str): WhisperX model name.
        language (str): Language code for WhisperX.
        batch_size (int): WhisperX batch size.
        device (str): Device for computation.
        compute_type (str): WhisperX compute type.
        preprocess_lowercase (bool): Lowercase text in preprocessing.
        preprocess_remove_punct (bool): Remove punctuation in preprocessing.
        preprocess_segment (bool): Segment text into sentences.
        mclip_model (str): M-CLIP model name.
        xlmr_model (str): XLM-R model name.
    """
    os.makedirs(output_dir, exist_ok=True)

    # 1. Transcribe audio to text
    print("Step 1: Transcribing audio with WhisperX...")
    whisperx_out = transcribe_with_whisperx(
        audio_path=input_audio,
        output_dir=output_dir,
        model_name=whisperx_model,
        language=language,
        device=device or ("cuda" if hasattr(sys, 'cuda') and sys.cuda else "cpu"),
        batch_size=batch_size,
        compute_type=compute_type or ("float16" if device == "cuda" else "float32")
    )
    transcript_txt = os.path.join(output_dir, "transcript.txt")
    if not os.path.exists(transcript_txt):
        raise RuntimeError(f"Transcript file not found: {transcript_txt}")

    # 2. Preprocess transcript
    print("Step 2: Preprocessing transcript...")
    preproc_txt = os.path.join(output_dir, "transcript_preprocessed.txt")
    preprocess_text_file(
        input_txt=transcript_txt,
        output_txt=preproc_txt,
        lowercase=preprocess_lowercase,
        remove_punct=preprocess_remove_punct,
        segment=preprocess_segment
    )

    # 3. Extract M-CLIP embeddings
    print("Step 3: Extracting M-CLIP embeddings...")
    mclip_npy = os.path.join(output_dir, "mclip_embeddings.npy")
    mclip_json = os.path.join(output_dir, "mclip_mapping.json")
    extract_mclip_embeddings(
        input_txt=preproc_txt,
        output_npy=mclip_npy,
        output_json=mclip_json,
        model_name=mclip_model,
        device=device
    )

    # 4. Extract XLM-R embeddings
    print("Step 4: Extracting XLM-R embeddings...")
    xlmr_npy = os.path.join(output_dir, "xlmr_embeddings.npy")
    xlmr_json = os.path.join(output_dir, "xlmr_mapping.json")
    extract_xlmr_embeddings(
        input_txt=preproc_txt,
        output_npy=xlmr_npy,
        output_json=xlmr_json,
        model_name=xlmr_model,
        device=device
    )
    print("Pipeline complete. All outputs saved in:", output_dir)

def preprocess_text_file(input_txt, output_txt, lowercase=True, remove_punct=True, segment=False):
    """
    Preprocesses a text file by cleaning and optionally segmenting sentences.
    Args:
        input_txt (str): Path to the input text file.
        output_txt (str): Path to save the cleaned text file.
        lowercase (bool): Whether to convert text to lowercase.
        remove_punct (bool): Whether to remove punctuation.
        segment (bool): Whether to split text into sentences.
    """
    import re
    import string
    def clean_text(text, lowercase=True, remove_punct=True):
        if lowercase:
            text = text.lower()
        if remove_punct:
            text = text.translate(str.maketrans('', '', string.punctuation))
        return text.strip()
    def segment_sentences(text):
        sentences = re.split(r'[.!?]+\s*', text)
        return [s.strip() for s in sentences if s.strip()]
    with open(input_txt, 'r', encoding='utf-8') as f:
        text = f.read()
    if segment:
        sentences = segment_sentences(text)
    else:
        sentences = text.splitlines()
    cleaned = [clean_text(s, lowercase, remove_punct) for s in sentences if s.strip()]
    with open(output_txt, 'w', encoding='utf-8') as f:
        for line in cleaned:
            f.write(line + '\n')
    print(f"Preprocessed text saved to: {output_txt}")

def main():
    parser = argparse.ArgumentParser(description="Semantic feature extraction pipeline: audio -> transcript -> embeddings.")
    parser.add_argument('--input_audio', type=str, default='data/012.wav', help='Input audio file (default: data/012.wav).')
    parser.add_argument('--output_dir', type=str, default='semantic_pipeline_output', help='Directory to save all outputs.')
    parser.add_argument('--no_lowercase', action='store_true', help='Do not lowercase text in preprocessing.')
    parser.add_argument('--no_remove_punct', action='store_true', help='Do not remove punctuation in preprocessing.')
    parser.add_argument('--segment', action='store_true', help='Segment text into sentences in preprocessing.')
    parser.add_argument('--device', type=str, default=None, help='Device for computation ("cuda" or "cpu").')
    args = parser.parse_args()

    pipeline(
        input_audio=args.input_audio,
        output_dir=args.output_dir,
        preprocess_lowercase=not args.no_lowercase,
        preprocess_remove_punct=not args.no_remove_punct,
        preprocess_segment=args.segment,
        device=args.device
    )

if __name__ == '__main__':
    main() 