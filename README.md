
---

## 📁 Project Structure

```

project_root/
├── datasets/                    # Dataset & preprocessing
│   ├── multimodal_gaze_dataset.py
│   └── preprocess_vgs.py
├── feature_extraction/          # Feature extraction scripts
│   ├── audio
│   │   ├── extract_egemaps.py
│   │   ├── extract_wav2vec.py
│   │   └── process_audio.py
│   ├── semantic
│   │   ├── extract_mclip.py
│   │   ├── extract_whisperx.py
│   │   └── extract_xlmr.py
│   └── visual
│       ├── extract_clip.py
│       ├── extract_dino.py
│       └── preprocess_images.py
├── models/                      # Model definitions
│   ├── multimodal_transformer.py
│   ├── gaze_head.py
│   └── full_model.py
├── utils/                       # Tools: loss, metrics, vis
│   ├── losses.py
│   ├── metrics.py
│   └── visualization.py
├── configs/                     # YAML config files
│   └── multimodal.yaml
├── outputs/                     # Checkpoints, logs, visual output
├── train.py                     # Training pipeline
├── evaluate.py                  # Evaluation script
├── README.md
└── environment.yml

```


---

## 🚧 Project Development TODO

### 🛠️ Feature Extraction

* \[✓] **Audio:** eGeMAPS and Wav2Vec2.0 extraction, windowed features saved as `.npy`
* \[✓] **Visual:** CLIP/DINOv2 features extraction, pre-processing, saving standardized features
* \[✓] **Semantic:** WhisperX transcription, XLM-R/M-CLIP embeddings, saving text features
* \[✓] **Multimodal Alignment:**
  Scripts for precise time/frame alignment of all modalities and ground truth labels
  *— All features and labels are aligned using the same sliding window parameters:*
      • Frame rate: **25 fps** (0.04 s per frame)
      • Window size: **1.0 s** (25 frames)
      • Step: **0.04 s** (1 frame)

---

### 🤝 Fusion Model

* \[✓] **Multi-modal Transformer fusion** (visual/audio/semantic token input)
* [ ] **Group-level attention features:** Add "who looks at whom" social tokens for advanced social gaze modeling
* [ ] **Cross-modal attention visualization and interpretability tools:** Visualize attention maps and improve model explainability

---

### 🚦 Training & Evaluation

* \[✓] **Basic train/eval pipeline** using aligned features
* [ ] **Batch/multi-clip training & efficient dataloading**
* [ ] **Evaluation metrics:** Add L2 distance, AUC, and other relevant metrics
* [ ] **Baseline & ablation experiments:** Compare with MMGaze, GazeLLE, and other published approaches

---

### 📚 Docs & Utilities

* \[✓] **Main README and example usage**
* [ ] **Code comments and per-script usage docs**
* [ ] **Config templates and demo visualizations**

---

## 🚀 Getting Started

```bash
# 1. Install dependencies
conda env create -f environment.yml
conda activate gazelle

# 2. Extract audio features
python feature_extraction/audio/process_audio.py \
  --input_audio_dir data/audio \
  --output_dir data/audio_feat

# 3. Extract text features (WhisperX → XLM-R + M-CLIP)
python feature_extraction/semantic/extract_whisperx.py \
  --input_audio_dir data/audio \
  --output_dir data/text_feat

# 4. Extract visual features (optional if using CLIP)
python feature_extraction/visual/extract_clip.py \
  --input_dir data/frames \
  --output_npy data/visual_feat.npy \
  --model_name 'ViT-B-16-plus-240' \
  --pretrained 'laion400m_e32'

# 5. Train the model
python train.py --config configs/multimodal.yaml

# 6. Evaluate the model
python evaluate.py --checkpoint outputs/best_model.pth
```

---

## 🛠️ Required Python Packages & Frameworks

### Core ML Framework

* `torch` (>= 2.6)
* `torchvision`
* `torchaudio`
* `transformers` (for Wav2Vec2, XLM-R, M-CLIP)
* `openai-whisper` / `whisperx`
* `open-clip-torch`
* `pytorch-lightning`
* `jsonargparse`

### Audio Processing

* `librosa`
* `opensmile`

### Visual Embedding

* `CLIP` (via `open-clip-torch` or `clip` from OpenAI)

### Data Handling / Utilities

* `numpy`
* `pandas`
* `tqdm`
* `scikit-learn`
* `timm`
* `matplotlib`
* `sentence-transformers`
* `sentencepiece`
* `opencv-python`
* `omegaconf` or `PyYAML` (for config management)

---
