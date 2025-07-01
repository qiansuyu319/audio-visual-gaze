"""
WhisperX Speech Transcription with Word-Level Timestamps

This script provides high-quality speech-to-text transcription using WhisperX,
which combines OpenAI's Whisper model with forced alignment to generate
precise word-level timestamps for each transcribed word.

Features:
- Automatic language detection
- Word-level timestamp alignment
- Multiple output formats (plain text and JSON)
- GPU acceleration support
- Configurable model sizes
"""
import argparse
import os
import json
import torch
import time

try:
    import whisperx
except ImportError:
    print("WhisperX not found. Please install it with:")
    print("pip install -U git+https://github.com/m-bain/whisperx.git")
    print("You also need ffmpeg installed on your system.")
    exit(1)

def transcribe_with_whisperx(
    audio_path,
    output_dir,
    model_name="large-v2",
    language=None,
    device="cuda",
    batch_size=16,
    compute_type="float16"
):
    """
    Transcribes an audio file using WhisperX with word-level timestamps.
    
    Args:
        audio_path (str): Path to the input audio file
        output_dir (str): Directory to save output files
        model_name (str): Whisper model name (tiny, base, large-v2, etc.)
        language (str): Language code (e.g., 'en', 'es') or None for auto-detection
        device (str): Device to use ('cuda' or 'cpu')
        batch_size (int): Batch size for transcription
        compute_type (str): Compute type for model quantization
        
    Returns:
        dict: The aligned transcription result with word-level timestamps
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Starting WhisperX transcription with model '{model_name}'...")
    start_time = time.time()
    
    # 1. Load WhisperX model
    print(f"Loading WhisperX model on device '{device}'...")
    model = whisperx.load_model(model_name, device, compute_type=compute_type)
    
    # 2. Load audio
    print(f"Loading audio from: {audio_path}")
    audio = whisperx.load_audio(audio_path)
    print(f"Audio loaded: {len(audio)} samples")
    
    # 3. Transcribe audio
    print("Transcribing audio...")
    result = model.transcribe(audio, batch_size=batch_size, language=language)
    print(f"Transcription complete. Detected language: {result['language']}")
    
    # 4. Align for word-level timestamps
    print("Performing word-level alignment...")
    align_model, metadata = whisperx.load_align_model(
        language_code=result["language"], 
        device=device
    )
    aligned_result = whisperx.align(
        result["segments"], 
        align_model, 
        metadata, 
        audio, 
        device, 
        return_char_alignments=False
    )
    print("Alignment complete.")
    
    # 5. Save outputs
    txt_path = os.path.join(output_dir, "transcript.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        for segment in aligned_result["segments"]:
            f.write(segment['text'].strip() + "\n")
    print(f"Plain text transcript saved to: {txt_path}")
    
    json_path = os.path.join(output_dir, "transcript_word_level.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(aligned_result, f, indent=2, ensure_ascii=False)
    print(f"Word-level JSON transcript saved to: {json_path}")
    
    # 6. Print summary
    total_words = sum(len(segment.get('words', [])) for segment in aligned_result["segments"])
    duration = time.time() - start_time
    print(f"\nTranscription Summary:")
    print(f"- Total segments: {len(aligned_result['segments'])}")
    print(f"- Total words: {total_words}")
    print(f"- Processing time: {duration:.2f} seconds")
    
    return aligned_result

def main():
    """Command-line interface for WhisperX transcription."""
    parser = argparse.ArgumentParser(
        description="Transcribe audio with word-level timestamps using WhisperX"
    )
    parser.add_argument(
        "--input_audio", 
        type=str, 
        required=True, 
        help="Path to input audio file"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        required=True, 
        help="Directory to save transcription files"
    )
    parser.add_argument(
        "--model", 
        type=str, 
        default="large-v2",
        help="Whisper model name (tiny, base, large-v2, etc.)"
    )
    parser.add_argument(
        "--language", 
        type=str, 
        default=None,
        help="Language code (e.g., 'en', 'es') or None for auto-detection"
    )
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=16,
        help="Batch size for transcription"
    )
    
    # Auto-detect device and compute type
    default_device = "cuda" if torch.cuda.is_available() else "cpu"
    default_compute_type = "float16" if torch.cuda.is_available() else "float32"
    
    parser.add_argument(
        "--device", 
        type=str, 
        default=default_device,
        help="Device for computation ('cuda' or 'cpu')"
    )
    parser.add_argument(
        "--compute_type", 
        type=str, 
        default=default_compute_type,
        choices=["float16", "int8", "float32"],
        help="Compute type for model quantization"
    )
    
    args = parser.parse_args()
    
    # Validate CPU settings
    if args.device == "cpu" and args.compute_type not in ["float32", "int8"]:
        print(f"Warning: Compute type '{args.compute_type}' not well-supported on CPU. Using 'float32'.")
        args.compute_type = "float32"
    
    try:
        transcribe_with_whisperx(
            audio_path=args.input_audio,
            output_dir=args.output_dir,
            model_name=args.model,
            language=args.language,
            device=args.device,
            batch_size=args.batch_size,
            compute_type=args.compute_type,
        )
    except Exception as e:
        print(f"Error during transcription: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main() 