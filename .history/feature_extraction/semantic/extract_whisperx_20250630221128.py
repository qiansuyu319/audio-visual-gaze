# WhisperX for Accurate Word-Level Timestamps
import argparse
import os
import json
import torch

try:
    import whisperx
except ImportError:
    print("WhisperX not found. Please install it with the following command:")
    print("\npip install -U git+https://github.com/m-bain/whisperx.git")
    print("\nYou also need to have ffmpeg installed on your system.")
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
    Transcribes an audio file using the WhisperX library to produce
    a transcript with word-level timestamps.

    Args:
        audio_path (str): Path to the input audio file.
        output_dir (str): Directory where the output files (.txt, .json) will be saved.
        model_name (str): The name of the Whisper model to use (e.g., "tiny", "base", "large-v2").
        language (str, optional): The language of the audio. If None, it will be auto-detected.
        device (str): The device to run the model on ("cuda" or "cpu").
        batch_size (int): The batch size for transcription to optimize memory usage.
        compute_type (str): The quantization type for the model (e.g., "float16", "int8").
    """
    os.makedirs(output_dir, exist_ok=True)

    # 1. Load the WhisperX model
    print(f"Loading WhisperX model '{model_name}' on device '{device}' with compute type '{compute_type}'...")
    model = whisperx.load_model(model_name, device, compute_type=compute_type)

    # 2. Load the audio file
    print(f"Loading audio from: {audio_path}")
    audio = whisperx.load_audio(audio_path)

    # 3. Transcribe the audio
    print("Transcribing audio...")
    result = model.transcribe(audio, batch_size=batch_size, language=language)
    print("Transcription complete.")

    # 4. Perform alignment to get word-level timestamps
    print("Aligning transcript to generate word-level timestamps...")
    align_model, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
    aligned_result = whisperx.align(result["segments"], align_model, metadata, audio, device, return_char_alignments=False)
    print("Alignment complete.")

    # 5. Save the outputs
    txt_path = os.path.join(output_dir, "transcript.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        for segment in aligned_result["segments"]:
            f.write(segment['text'].strip() + "\n")
    print(f"Plain text transcript saved to: {txt_path}")

    json_path = os.path.join(output_dir, "transcript_word_level.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(aligned_result, f, indent=2, ensure_ascii=False)
    print(f"Word-level JSON transcript saved to: {json_path}")
    
    return aligned_result

def main():
    """Defines the command-line interface and runs the transcription process."""
    parser = argparse.ArgumentParser(description="Transcribe an audio file with word-level timestamps using WhisperX.")
    parser.add_argument("--input_audio", type=str, required=True, help="Path to the input audio file (e.g., .wav, .mp3).")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the transcription files. 推荐放在 output/semantic_feature/")
    parser.add_argument("--model", type=str, default="large-v2", help="Name of the Whisper model to use (e.g., 'tiny', 'base', 'large-v2').")
    parser.add_argument("--language", type=str, default=None, help="Two-letter language code (e.g., 'en', 'es'). If not specified, WhisperX will auto-detect.")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for transcription on GPU.")
    
    # Automatically select device and a compatible compute type
    default_device = "cuda" if torch.cuda.is_available() else "cpu"
    default_compute_type = "float16" if torch.cuda.is_available() else "float32"
    
    parser.add_argument("--device", type=str, default=default_device, help="Device for computation ('cuda' or 'cpu').")
    parser.add_argument("--compute_type", type=str, default=default_compute_type, choices=["float16", "int8", "float32"], help="Compute type for the model.")

    args = parser.parse_args()

    # Validate settings for CPU
    if args.device == "cpu" and args.compute_type not in ["float32", "int8"]:
        print(f"Warning: Compute type '{args.compute_type}' is not well-supported on CPU. Defaulting to 'float32'.")
        args.compute_type = "float32"

    transcribe_with_whisperx(
        audio_path=args.input_audio,
        output_dir=args.output_dir,
        model_name=args.model,
        language=args.language,
        device=args.device,
        batch_size=args.batch_size,
        compute_type=args.compute_type,
    )

if __name__ == '__main__':
    main()