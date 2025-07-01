import subprocess

if __name__ == '__main__':
    print('Testing extract_egemaps.py...')
    subprocess.run(['python', 'test_extract_egemaps.py'], check=True)
    print('Testing extract_wav2vec.py...')
    subprocess.run(['python', 'test_extract_wav2vec.py'], check=True)
    print('Testing preprocess_audio.py...')
    subprocess.run(['python', 'test_preprocess_audio.py'], check=True)
    print('All audio feature extraction tests passed!') 