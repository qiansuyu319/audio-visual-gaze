Great — since your project is clearly a **large-model-based research system** (multimodal gaze prediction using PyTorch, leveraging audio, vision, and semantic embeddings), here's a clean, precise **English-style engineering rule document** tailored specifically for your setup.

---

# 🧠 Engineering Rules for Multimodal Gaze Target Prediction (Python, PyTorch)

This document defines development conventions for implementing a research-level multimodal gaze prediction system using Python and PyTorch. It ensures consistency, readability, reproducibility, and maintainability in the context of deep learning and large-model experimentation.

---

## 📁 Project Structure

Use a modular, responsibility-driven directory layout:

```
project_root/
├── datasets/                    # Data loaders, VGS alignment, caching
├── features/                    # Feature extraction (audio, text, visual)
├── models/                      # Architecture, fusion modules, prediction heads
├── utils/                       # Loss functions, metrics, visualization
├── configs/                     # YAML or JSON config files
├── outputs/                     # Saved weights, logs, visual results
├── train.py                     # Main training pipeline
├── evaluate.py                  # Evaluation script
└── requirements.txt             # Environment dependencies
```

---

## 🧠 Coding Style (Python)

### ✅ General

* Use **functional, declarative programming** where possible.
* Prefer **stateless helpers** over class-heavy design.
* Use **PyTorch modules (`nn.Module`)** only where needed for trainable components.

### ✅ Naming Conventions

| Concept     | Convention                                                  |
| ----------- | ----------------------------------------------------------- |
| Variables   | `audio_feat`, `fused_feat`, `inout_flag`                    |
| Booleans    | Use auxiliary verbs: `is_valid`, `has_error`, `should_skip` |
| File names  | snake\_case, e.g. `multimodal_transformer.py`               |
| Config keys | kebab-case or snake\_case                                   |

### ✅ Function Signatures

Use RORO (Receive an Object, Return an Object) style:

```python
def forward(*, image: Tensor, audio_feat: Tensor, text_feat: Tensor) -> dict:
    ...
    return {"heatmap": heatmap, "inout": inout_logits}
```

---

## ⚙️ Training Conventions

### ✅ Error Handling

* Use **guard clauses** and **early return** patterns:

  ```python
  if gaze_target is None:
      return torch.zeros_like(default_map)
  ```
* Handle all unexpected conditions explicitly. Raise errors with context:

  ```python
  raise ValueError(f"Missing label for frame {frame_id}")
  ```

### ✅ Loss Design

* Always use weighted multi-task loss:

  ```python
  loss = heatmap_loss + lambda_inout * inout_loss
  ```

* Define loss weights in config, not code.

---

## 🔁 Feature Extraction

* Always **extract features offline** (Wav2Vec2, XLM-R, CLIP, M-CLIP).
* Align all features per frame using timestamps.
* Save each aligned sample as `.pt` or `.npy` with standardized naming:
  `video_{id}_frame_{index}.pt`

---

## 🧪 Evaluation

* Primary metrics:

  * `l2_distance(pred_coords, gt_coords)`
  * `inout_accuracy(pred_logits, inout_gt)`

* Optional:

  * AUC (for heatmap confidence)
  * Hit\@1 / Hit\@5 for top-k gaze prediction

* Save visual output:

  * Overlay heatmap + ground-truth for inspection
  * Filename format: `viz_{video_id}_{frame_id}.png`

---

## 📜 Configuration

* Use centralized config files (`configs/multimodal.yaml`) to define:

  * Model parameters
  * Paths to features, labels, outputs
  * Training hyperparameters
  * Modalities to enable/disable (`use_audio`, `use_text`...)

---

## 💾 File I/O and Logging

* Use `tqdm` for progress bars
* Use `logging` module or `rich` for structured logs
* Save:

  * Best model: `outputs/best_model.pt`
  * Last checkpoint: `outputs/latest.pt`
  * Metrics: `outputs/metrics.json`
  * Visualizations: `outputs/viz/*.png`

---

## 🛠️ Environment & Dependencies

* Python ≥ 3.9
* PyTorch ≥ 1.12
* Dependencies managed in `requirements.txt`
* Use virtualenv or conda environment
* Recommend `torch.cuda.amp` for mixed-precision training

---

## 🧬 Reproducibility

* Fix all random seeds:

  ```python
  torch.manual_seed(seed)
  np.random.seed(seed)
  random.seed(seed)
  ```
* Use `deterministic=True` in `torch.backends.cudnn`

---

## 🧰 File Naming Summary

| File/Module                  | Purpose                              |
| ---------------------------- | ------------------------------------ |
| `extract_audio_features.py`  | Extract Wav2Vec2, eGeMAPS            |
| `extract_text_features.py`   | WhisperX + XLM-R + M-CLIP            |
| `multimodal_gaze_dataset.py` | Loads frame-aligned multimodal input |
| `multimodal_transformer.py`  | Model encoder with fused inputs      |
| `gaze_head.py`               | Heatmap + in/out prediction head     |
| `train.py`                   | Training entry point                 |
| `evaluate.py`                | Evaluation and metric logging        |

---

## 🧑‍🔬 Authoring Best Practices

* Every `model/*.py` should export a `build_model(config)` function.
* Every `feature/*.py` should be runnable as a CLI script and save to `features/`.
* All outputs (models, logs, metrics, visuals) go in `outputs/`.

---

Let me know if you'd like this turned into a downloadable Markdown file or committed to your project directly as `docs/engineering-rules.md`.
