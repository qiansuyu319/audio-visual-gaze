import subprocess

def test_extract_whisperx():
    result = subprocess.run(['python', 'extract_whisperx.py'], capture_output=True, text=True)
    assert 'WhisperX' in result.stdout or result.returncode == 0

if __name__ == '__main__':
    test_extract_whisperx()
    print('extract_whisperx.py test passed.') 