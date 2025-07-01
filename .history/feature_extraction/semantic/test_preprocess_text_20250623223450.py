import subprocess

def test_preprocess_text():
    result = subprocess.run(['python', 'preprocess_text.py'], capture_output=True, text=True)
    assert 'Preprocessing text' in result.stdout or result.returncode == 0

if __name__ == '__main__':
    test_preprocess_text()
    print('preprocess_text.py test passed.') 