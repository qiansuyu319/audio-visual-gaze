import subprocess

def test_extract_xlmr():
    result = subprocess.run(['python', 'extract_xlmr.py'], capture_output=True, text=True)
    assert 'XLM-R' in result.stdout or result.returncode == 0

if __name__ == '__main__':
    test_extract_xlmr()
    print('extract_xlmr.py test passed.') 