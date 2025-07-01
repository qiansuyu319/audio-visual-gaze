import subprocess
import os
import pytest

def test_preprocess_text_success():
    output_dir = 'semantic_pipeline_output_test'
    input_audio = 'data/012.wav'
    # Remove output dir if exists
    if os.path.exists(output_dir):
        import shutil
        shutil.rmtree(output_dir)
    result = subprocess.run([
        'python', 'preprocess_text.py',
        '--input_audio', input_audio,
        '--output_dir', output_dir
    ], capture_output=True, text=True)
    assert result.returncode == 0, f"Pipeline failed: {result.stderr}"
    # Check output files
    assert os.path.exists(os.path.join(output_dir, 'transcript.txt'))
    assert os.path.exists(os.path.join(output_dir, 'transcript_preprocessed.txt'))
    assert os.path.exists(os.path.join(output_dir, 'mclip_embeddings.npy'))
    assert os.path.exists(os.path.join(output_dir, 'xlmr_embeddings.npy'))

def test_preprocess_text_missing_audio():
    output_dir = 'semantic_pipeline_output_test_missing'
    input_audio = 'data/nonexistent.wav'
    result = subprocess.run([
        'python', 'preprocess_text.py',
        '--input_audio', input_audio,
        '--output_dir', output_dir
    ], capture_output=True, text=True)
    assert result.returncode != 0
    assert 'not found' in result.stderr.lower() or 'no such file' in result.stderr.lower() or 'error' in result.stderr.lower()

def test_preprocess_text_no_lowercase():
    output_dir = 'semantic_pipeline_output_test_nolower'
    input_audio = 'data/012.wav'
    if os.path.exists(output_dir):
        import shutil
        shutil.rmtree(output_dir)
    result = subprocess.run([
        'python', 'preprocess_text.py',
        '--input_audio', input_audio,
        '--output_dir', output_dir,
        '--no_lowercase'
    ], capture_output=True, text=True)
    assert result.returncode == 0
    # Check that the preprocessed text is not all lowercase
    preproc_txt = os.path.join(output_dir, 'transcript_preprocessed.txt')
    assert os.path.exists(preproc_txt)
    with open(preproc_txt, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        assert any(any(c.isupper() for c in line) for line in lines)

def test_preprocess_text_no_remove_punct():
    output_dir = 'semantic_pipeline_output_test_nopunct'
    input_audio = 'data/012.wav'
    if os.path.exists(output_dir):
        import shutil
        shutil.rmtree(output_dir)
    result = subprocess.run([
        'python', 'preprocess_text.py',
        '--input_audio', input_audio,
        '--output_dir', output_dir,
        '--no_remove_punct'
    ], capture_output=True, text=True)
    assert result.returncode == 0
    # Check that the preprocessed text contains punctuation
    preproc_txt = os.path.join(output_dir, 'transcript_preprocessed.txt')
    assert os.path.exists(preproc_txt)
    with open(preproc_txt, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        assert any(any(c in line for c in '.,!?') for line in lines)

def test_preprocess_text_segment():
    output_dir = 'semantic_pipeline_output_test_segment'
    input_audio = 'data/012.wav'
    if os.path.exists(output_dir):
        import shutil
        shutil.rmtree(output_dir)
    result = subprocess.run([
        'python', 'preprocess_text.py',
        '--input_audio', input_audio,
        '--output_dir', output_dir,
        '--segment'
    ], capture_output=True, text=True)
    assert result.returncode == 0
    # Check that the preprocessed text has more lines than the original transcript
    transcript_txt = os.path.join(output_dir, 'transcript.txt')
    preproc_txt = os.path.join(output_dir, 'transcript_preprocessed.txt')
    assert os.path.exists(transcript_txt)
    assert os.path.exists(preproc_txt)
    with open(transcript_txt, 'r', encoding='utf-8') as f:
        orig_lines = f.readlines()
    with open(preproc_txt, 'r', encoding='utf-8') as f:
        seg_lines = f.readlines()
    assert len(seg_lines) >= len(orig_lines)

if __name__ == '__main__':
    import sys
    import pytest
    sys.exit(pytest.main([__file__])) 