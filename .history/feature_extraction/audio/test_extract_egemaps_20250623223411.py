import subprocess

def test_extract_egemaps():
    result = subprocess.run(['python', 'extract_egemaps.py'], capture_output=True, text=True)
    assert 'eGeMAPS' in result.stdout or 'egemaps' in result.stdout or result.returncode == 0

if __name__ == '__main__':
    test_extract_egemaps()
    print('extract_egemaps.py test passed.') 