import subprocess

def test_extract_dino():
    result = subprocess.run(['python', 'extract_dino.py'], capture_output=True, text=True)
    assert 'DINOv2' in result.stdout or result.returncode == 0

if __name__ == '__main__':
    test_extract_dino()
    print('extract_dino.py test passed.') 