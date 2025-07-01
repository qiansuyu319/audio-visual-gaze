import subprocess

def test_extract_mclip():
    result = subprocess.run(['python', 'extract_mclip.py'], capture_output=True, text=True)
    assert 'M-CLIP' in result.stdout or result.returncode == 0

if __name__ == '__main__':
    test_extract_mclip()
    print('extract_mclip.py test passed.') 