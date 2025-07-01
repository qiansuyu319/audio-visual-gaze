# Text Preprocessing Script
# This script performs basic text cleaning and segmentation for NLP tasks.
# It provides a command-line interface for specifying input and output files and preprocessing options.

import argparse
import re
import string


def clean_text(text, lowercase=True, remove_punct=True):
    """
    Cleans a single text string by lowercasing and removing punctuation.

    Args:
        text (str): The input text string.
        lowercase (bool): Whether to convert text to lowercase.
        remove_punct (bool): Whether to remove punctuation.
    Returns:
        str: The cleaned text string.
    """
    if lowercase:
        text = text.lower()
    if remove_punct:
        text = text.translate(str.maketrans('', '', string.punctuation))
    return text.strip()


def segment_sentences(text):
    """
    Splits text into sentences using simple punctuation rules.
    Args:
        text (str): The input text string.
    Returns:
        list: List of sentence strings.
    """
    # Split on period, exclamation, or question mark followed by space or end of string
    sentences = re.split(r'[.!?]+\s*', text)
    return [s.strip() for s in sentences if s.strip()]


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
    parser = argparse.ArgumentParser(description="Preprocess a text file (cleaning, segmentation).")
    parser.add_argument('--input_txt', type=str, required=True, help='Input text file.')
    parser.add_argument('--output_txt', type=str, required=True, help='Output cleaned text file.')
    parser.add_argument('--no_lowercase', action='store_true', help='Do not lowercase text.')
    parser.add_argument('--no_remove_punct', action='store_true', help='Do not remove punctuation.')
    parser.add_argument('--segment', action='store_true', help='Segment text into sentences.')
    args = parser.parse_args()

    preprocess_text_file(
        input_txt=args.input_txt,
        output_txt=args.output_txt,
        lowercase=not args.no_lowercase,
        remove_punct=not args.no_remove_punct,
        segment=args.segment
    )

if __name__ == '__main__':
    main() 