# Visual Modality Features

- **Common Feature Dimensions**: e.g. patch token (1024), head token (768/1024), etc.
- **File Format**: `.npy` (NumPy array), shape `[num_frames, N1, visual_dim]` or `[N1, visual_dim]` per file.
- **Alignment**: Each file corresponds to a video segment or window, filenames should match across modalities for alignment (e.g., `window_00000_clip.npy`).
- **Extraction Script**: See `extract_clip.py`, `extract_dino.py`.

> For new visual features, add new extract_*.py scripts and document here. 