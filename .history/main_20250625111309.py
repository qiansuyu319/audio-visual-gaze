import argparse
import os
import subprocess

def run_script(script_path, args):
    command = ['python', script_path] + args
    print(f"Running command: {' '.join(command)}")
    subprocess.run(command, check=True)

def main():
    parser = argparse.ArgumentParser(description="Run the full feature extraction pipeline.")
    parser.add_argument('--input_dir', type=str, default='data', help='Input directory for data.')
    parser.add_argument('--output_dir', type=str, default='processed_data', help='Output directory for features.')
    parser.add_argument('--model_name', type=str, default='ViT-B-16-plus-240', help='Model name for open_clip.')
    parser.add_argument('--pretrained', type=str, default='laion400m_e32', help='Pretrained weights for open_clip.')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Audio feature extraction
    audio_input_dir = os.path.join(args.input_dir, 'audio')
    audio_output_dir = os.path.join(args.output_dir, 'audio_features')
    os.makedirs(audio_output_dir, exist_ok=True)
    run_script('feature_extraction/audio/process_audio.py', [
        '--input_audio_dir', audio_input_dir,
        '--output_dir', audio_output_dir
    ])

    # Semantic feature extraction
    semantic_output_dir = os.path.join(args.output_dir, 'semantic_features')
    os.makedirs(semantic_output_dir, exist_ok=True)
    run_script('feature_extraction/semantic/extract_whisperx.py', [
        '--input_audio_dir', audio_input_dir,
        '--output_dir', semantic_output_dir
    ])

    # Visual feature extraction
    visual_input_dir = os.path.join(args.input_dir, 'frames')
    visual_output_npy = os.path.join(args.output_dir, 'visual_features.npy')
    run_script('feature_extraction/visual/extract_clip.py', [
        '--input_dir', visual_input_dir,
        '--output_npy', visual_output_npy,
        '--model_name', args.model_name,
        '--pretrained', args.pretrained
    ])

if __name__ == "__main__":
    main() 