import subprocess

def test_extract_clip():
    result = subprocess.run(['python', 'extract_clip.py'], capture_output=True, text=True)
    assert 'CLIP' in result.stdout or result.returncode == 0

if __name__ == '__main__':
    test_extract_clip()
    print('extract_clip.py test passed.') 