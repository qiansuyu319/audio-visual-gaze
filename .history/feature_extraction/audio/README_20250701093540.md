# Audio Modality Features

- **Common Feature Dimensions**: e.g. wav2vec (768), egemaps (88), etc.
- **File Format**: `.npy` (NumPy array), shape `[num_frames, N2, audio_dim]` or `[N2, audio_dim]` per file.
- **Alignment**: Each file corresponds to a video segment or window, filenames should match across modalities for alignment (e.g., `window_00000_wav2vec.npy`).
- **Extraction Script**: See `extract_wav2vec.py`, `extract_egemaps.py`.

> For new audio features, add new extract_*.py scripts and document here. 