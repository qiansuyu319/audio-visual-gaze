import subprocess
import tempfile
import numpy as np
import soundfile as sf
import os

def test_preprocess_audio():
    # Create a dummy wav file
    with tempfile.TemporaryDirectory() as tmpdir:
        wav_path = os.path.join(tmpdir, 'test.wav')
        out_dir = os.path.join(tmpdir, 'chunks')
        # 3 seconds of silence at 16kHz
        sf.write(wav_path, np.zeros(16000 * 3), 16000)
        result = subprocess.run(['python', 'preprocess_audio.py', '--input_wav', wav_path, '--output_dir', out_dir, '--chunk_length', '1.0'], capture_output=True, text=True)
        files = os.listdir(out_dir)
        assert any(f.endswith('.wav') for f in files)
        print('preprocess_audio.py test passed.')

if __name__ == '__main__':
    test_preprocess_audio() 