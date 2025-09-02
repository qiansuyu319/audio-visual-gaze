"""
CPU-only WhisperX wrapper that completely avoids CUDA initialization
This module provides a safe way to use WhisperX on CPU without CUDA conflicts
"""
import os
import sys
import subprocess
import tempfile
import json
import torch


def force_cpu_environment():
    """Set environment variables to force CPU-only execution"""
    env_vars = {
        'CUDA_VISIBLE_DEVICES': '',
        'CUDNN_PATH': '',
        'LD_LIBRARY_PATH': '',
        'PYTORCH_CUDA_ALLOC_CONF': '',
        'CUDA_LAUNCH_BLOCKING': '1',
        'OMP_NUM_THREADS': '1',
        'MKL_NUM_THREADS': '1',
        'FORCE_CPU': '1'
    }
    
    for key, value in env_vars.items():
        os.environ[key] = value


def cpu_only_transcribe(audio_path, model_name="base", language=None, batch_size=4):
    """
    Transcribe audio using WhisperX in a separate process with CPU-only environment
    
    Args:
        audio_path (str): Path to audio file
        model_name (str): WhisperX model name (default: "base" for CPU)
        language (str): Language code (None for auto-detection)
        batch_size (int): Batch size (smaller for CPU)
    
    Returns:
        dict: Transcription result or None if failed
    """
    # Create temporary directory for output
    with tempfile.TemporaryDirectory() as temp_dir:
        output_path = os.path.join(temp_dir, 'transcript_word_level.json')
        
        # Prepare environment for subprocess
        env = os.environ.copy()
        env.update({
            'CUDA_VISIBLE_DEVICES': '',
            'CUDNN_PATH': '',
            'LD_LIBRARY_PATH': '',
            'PYTORCH_CUDA_ALLOC_CONF': '',
            'CUDA_LAUNCH_BLOCKING': '1',
            'OMP_NUM_THREADS': '1',
            'MKL_NUM_THREADS': '1',
            'FORCE_CPU': '1'
        })
        
        # Build command
        cmd = [
            sys.executable, '-m', 'feature_extraction.semantic.extract_whisperx',
            '--input_audio', audio_path,
            '--output_dir', temp_dir,
            '--model', model_name,
            '--device', 'cpu',
            '--compute_type', 'float32',
            '--batch_size', str(batch_size)
        ]
        
        if language:
            cmd.extend(['--language', language])
        
        try:
            print(f"🔄 Running CPU-only WhisperX transcription...")
            result = subprocess.run(
                cmd,
                env=env,
                capture_output=True,
                text=True,
                timeout=600  # 10 minute timeout
            )
            
            if result.returncode == 0:
                # Load the result
                if os.path.exists(output_path):
                    with open(output_path, 'r', encoding='utf-8') as f:
                        return json.load(f)
                else:
                    print(f"❌ Output file not found: {output_path}")
                    return None
            else:
                print(f"❌ WhisperX subprocess failed:")
                print(f"STDOUT: {result.stdout}")
                print(f"STDERR: {result.stderr}")
                return None
                
        except subprocess.TimeoutExpired:
            print("❌ WhisperX transcription timed out")
            return None
        except Exception as e:
            print(f"❌ Error running WhisperX subprocess: {e}")
            return None


def safe_import_whisperx():
    """
    Safely import WhisperX with CPU-only environment
    Returns the whisperx module or None if import fails
    """
    try:
        # Force CPU environment before import
        force_cpu_environment()
        
        # Import with CPU-only torch backend
        import torch
        if torch.cuda.is_available():
            # Force torch to ignore CUDA
            torch.cuda.is_available = lambda: False
            torch.cuda.device_count = lambda: 0
        
        import whisperx
        return whisperx
    except Exception as e:
        print(f"❌ Failed to import WhisperX safely: {e}")
        return None


def cpu_whisperx_transcribe(audio_path, output_dir, model_name="base", language=None, batch_size=4):
    """
    Direct CPU-only WhisperX transcription with environment isolation
    
    Args:
        audio_path (str): Path to audio file
        output_dir (str): Output directory for results
        model_name (str): WhisperX model name
        language (str): Language code (None for auto-detection)
        batch_size (int): Batch size for processing
    
    Returns:
        dict: Transcription result or None if failed
    """
    # Force CPU environment
    force_cpu_environment()
    
    # Try to import WhisperX safely
    whisperx = safe_import_whisperx()
    if whisperx is None:
        print("❌ Could not safely import WhisperX")
        return None
    
    try:
        print(f"🔄 Loading WhisperX model '{model_name}' on CPU...")
        
        # Load model with explicit CPU settings
        model = whisperx.load_model(
            model_name, 
            device="cpu", 
            compute_type="float32"
        )
        
        print(f"🎵 Loading audio: {audio_path}")
        audio = whisperx.load_audio(audio_path)
        
        print("🎙️ Transcribing with CPU...")
        result = model.transcribe(
            audio, 
            batch_size=batch_size, 
            language=language
        )
        
        if not result or 'segments' not in result:
            print("❌ No transcription segments found")
            return None
        
        print(f"✅ Transcription complete! Language: {result.get('language', 'unknown')}")
        
        # Save results
        os.makedirs(output_dir, exist_ok=True)
        
        # Save text transcript
        txt_path = os.path.join(output_dir, "transcript.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            for segment in result["segments"]:
                f.write(segment['text'].strip() + "\n")
        
        # Save JSON transcript
        json_path = os.path.join(output_dir, "transcript_word_level.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Results saved to: {output_dir}")
        return result
        
    except Exception as e:
        print(f"❌ CPU WhisperX transcription failed: {e}")
        return None
