import subprocess
import tempfile
import numpy as np
import soundfile as sf
import os

def test_extract_wav2vec():
    # Create a dummy wav file
    with tempfile.TemporaryDirectory() as tmpdir:
        wav_path = os.path.join(tmpdir, 'test.wav')
        out_path = os.path.join(tmpdir, 'out.npy')
        # 1 second of silence at 16kHz
        sf.write(wav_path, np.zeros(16000), 16000)
        result = subprocess.run(['python', 'extract_wav2vec.py', '--input_wav', wav_path, '--output_npy', out_path], capture_output=True, text=True)
        assert os.path.exists(out_path)
        print('extract_wav2vec.py test passed.')

if __name__ == '__main__':
    test_extract_wav2vec() 