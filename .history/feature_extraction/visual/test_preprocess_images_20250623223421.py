import subprocess

def test_preprocess_images():
    result = subprocess.run(['python', 'preprocess_images.py'], capture_output=True, text=True)
    assert 'Preprocessing images' in result.stdout or result.returncode == 0

if __name__ == '__main__':
    test_preprocess_images()
    print('preprocess_images.py test passed.') 