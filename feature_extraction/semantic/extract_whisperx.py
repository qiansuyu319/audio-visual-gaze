# WhisperX for Accurate Word-Level Timestamps
import argparse
import os
import json
import torch
import omegaconf.listconfig
import omegaconf.base
import time
from tqdm import tqdm

# torch.serialization.add_safe_globals not available in PyTorch 2.3.1
# torch.serialization.add_safe_globals([
#     omegaconf.listconfig.ListConfig,
#     omegaconf.base.ContainerMetadata
# ])

import whisperx


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
    a transcript with word-level timestamps with progress bars.

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
    
    # Overall progress tracking
    print("🚀 Starting WhisperX Transcription Pipeline")
    print(f"📁 Audio file: {os.path.basename(audio_path)}")
    print(f"📂 Output directory: {output_dir}")
    print(f"🤖 Model: {model_name} | Device: {device} | Compute: {compute_type}")
    print("=" * 70)

    # Add error handling and memory management
    try:
        # Bus error prevention: Set conservative memory settings
        if device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
            # Disable TF32 to prevent reproducibility warnings and potential memory issues
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            print(f"GPU memory before loading: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
            
            # Check GPU memory availability
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            if gpu_memory < 4.0:  # Less than 4GB
                print("⚠️ Limited GPU memory detected, falling back to CPU for stability")
                device = "cpu"
                compute_type = "float32"

        # 1. Load the WhisperX model with progress bar
        print(f"🔄 Loading WhisperX model '{model_name}' on device '{device}' with compute type '{compute_type}'...")
        with tqdm(total=100, desc="Loading Model", unit="%", colour="blue") as pbar:
            model = whisperx.load_model(model_name, device, compute_type=compute_type)
            pbar.update(100)
        
        if device == "cuda" and torch.cuda.is_available():
            print(f"GPU memory after model loading: {torch.cuda.memory_allocated()/1024**3:.2f} GB")

        # 2. Load the audio file with progress bar and validation
        print(f"🎵 Loading audio from: {audio_path}")
        with tqdm(total=100, desc="Loading Audio", unit="%", colour="green") as pbar:
            # Check if audio file exists and is readable
            if not os.path.exists(audio_path):
                raise FileNotFoundError(f"Audio file not found: {audio_path}")
            
            # Load audio with error handling
            try:
                audio = whisperx.load_audio(audio_path)
                pbar.update(50)
                
                # Validate audio
                if len(audio) == 0:
                    raise ValueError("Audio file is empty or corrupted")
                
                # Check audio duration and adjust batch size if needed
                duration = len(audio) / 16000  # Assuming 16kHz sample rate
                print(f"🎵 Audio duration: {duration:.2f} seconds")
                
                if duration < 30:
                    print("⚠️ Audio shorter than 30s - using smaller batch size for stability")
                    batch_size = min(batch_size, 4)
                
                if duration < 5:
                    print("⚠️ Very short audio - forcing language to avoid detection issues")
                    if language is None:
                        language = "en"  # Default to English for short audio
                
                pbar.update(50)
                
            except Exception as audio_error:
                raise RuntimeError(f"Failed to load audio: {audio_error}")

        # 3. Transcribe the audio with progress bar
        print("🎙️ Transcribing audio (this may take a while for long audio files)...")
        with tqdm(total=100, desc="Transcribing", unit="%", colour="yellow") as pbar:
            # Start transcription
            pbar.set_description("Transcribing audio segments")
            result = model.transcribe(audio, batch_size=batch_size, language=language)
            pbar.update(100)
        print(f"✅ Transcription complete! Language detected: {result.get('language', 'unknown')}")

        # Check if transcription result is valid
        if not result or 'segments' not in result or not result['segments']:
            print("❌ Transcription failed: No segments found in result")
            return None

        # 4. Perform alignment to get word-level timestamps with progress bar
        print("🔗 Aligning transcript to generate word-level timestamps...")
        with tqdm(total=100, desc="Alignment", unit="%", colour="magenta") as pbar:
            pbar.set_description("Loading alignment model")
            pbar.update(30)
            try:
                # Check if alignment is supported for the detected language
                detected_lang = result.get("language", "en")
                print(f"🌐 Detected language: {detected_lang}")
                
                # Force CPU for alignment to avoid GPU memory issues with pyannote
                alignment_device = "cpu" if device == "cuda" else device
                if alignment_device != device:
                    print("🔄 Using CPU for alignment to avoid GPU memory issues")
                
                align_model, metadata = whisperx.load_align_model(
                    language_code=detected_lang, 
                    device=alignment_device
                )
                pbar.set_description("Performing alignment")
                pbar.update(30)
                
                # Perform alignment with error handling
                aligned_result = whisperx.align(
                    result["segments"], 
                    align_model, 
                    metadata, 
                    audio, 
                    alignment_device, 
                    return_char_alignments=False
                )
                pbar.update(40)
                
            except Exception as e:
                print(f"⚠️ Alignment failed: {e}")
                print("💡 This is often due to pyannote compatibility issues")
                print("🔄 Using transcription result without word-level alignment")
                aligned_result = result
                
                # Clean up any partial alignment models
                try:
                    if 'align_model' in locals():
                        del align_model
                    if 'metadata' in locals():
                        del metadata
                    if device == "cuda":
                        torch.cuda.empty_cache()
                except:
                    pass
                    
        print("✅ Alignment complete.")

        # 5. Save the outputs with progress bar
        print("💾 Saving transcription results...")
        with tqdm(total=100, desc="Saving Files", unit="%", colour="cyan") as pbar:
            # Save plain text transcript
            pbar.set_description("Saving text transcript")
            txt_path = os.path.join(output_dir, "transcript.txt")
            with open(txt_path, "w", encoding="utf-8") as f:
                for segment in aligned_result["segments"]:
                    f.write(segment['text'].strip() + "\n")
            pbar.update(50)
            
            # Save JSON with word-level timestamps
            pbar.set_description("Saving JSON transcript")
            json_path = os.path.join(output_dir, "transcript_word_level.json")
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(aligned_result, f, indent=2, ensure_ascii=False)
            pbar.update(50)
        
        print(f"✅ Plain text transcript saved to: {txt_path}")
        print(f"✅ Word-level JSON transcript saved to: {json_path}")
        
        # Print summary statistics
        total_segments = len(aligned_result["segments"])
        total_words = sum(len(segment.get("words", [])) for segment in aligned_result["segments"])
        print(f"📊 Transcription Summary: {total_segments} segments, {total_words} words processed")
        print("=" * 70)
        print("🎉 WhisperX transcription pipeline completed successfully!")
        
        return aligned_result
    
    except Exception as e:
        print(f"❌ Error loading WhisperX model: {e}")
        if device == "cuda":
            print("🔄 Falling back to CPU...")
            try:
                device = "cpu"
                compute_type = "float32"
                model = whisperx.load_model(model_name, device, compute_type=compute_type)
                
                # Retry the entire process with CPU
                print(f"🎵 Loading audio from: {audio_path}")
                audio = whisperx.load_audio(audio_path)
                
                print("🎙️ Transcribing audio with CPU...")
                result = model.transcribe(audio, batch_size=batch_size, language=language)
                
                if not result or 'segments' not in result or not result['segments']:
                    print("❌ CPU transcription also failed: No segments found")
                    return None
                
                print(f"✅ CPU Transcription complete! Language detected: {result.get('language', 'unknown')}")
                
                # Save outputs
                txt_path = os.path.join(output_dir, "transcript.txt")
                with open(txt_path, "w", encoding="utf-8") as f:
                    for segment in result["segments"]:
                        f.write(segment['text'].strip() + "\n")
                
                json_path = os.path.join(output_dir, "transcript_word_level.json")
                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(result, f, indent=2, ensure_ascii=False)
                
                print(f"✅ Files saved to: {txt_path}, {json_path}")
                return result
                
            except Exception as cpu_e:
                print(f"❌ CPU fallback also failed: {cpu_e}")
                return None
        else:
            print(f"❌ Transcription failed: {e}")
            return None


    
    finally:
        # Clean up GPU memory
        if device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("🧹 GPU memory cleaned up")

def main():
    """Defines the command-line interface and runs the transcription process."""
    parser = argparse.ArgumentParser(description="Transcribe an audio file with word-level timestamps using WhisperX.")
    parser.add_argument("--input_audio", type=str, required=True, help="Path to the input audio file (e.g., .wav, .mp3).")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the transcription files.")
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