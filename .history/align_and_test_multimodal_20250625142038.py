import numpy as np
import torch
from models.full_model import MultiModalFusion

# === File paths (edit as needed) ===
visual_path = 'test_data/output_features.npy'  # shape: [N, visual_dim]
audio_path = 'processed_audio_output/window_00000_wav2vec.npy'  # shape: [audio_dim] or [T, audio_dim]
semantic_path = 'semantic_pipeline_output/mclip_embeddings.npy'  # shape: [N, semantic_dim]

# === Load features ===
visual_feat = np.load(visual_path)  # [N, visual_dim]
audio_feat = np.load(audio_path)    # [audio_dim] or [T, audio_dim]
semantic_feat = np.load(semantic_path)  # [N, semantic_dim]

# === Select the first sample (index 0) ===
visual_sample = visual_feat[0]  # [visual_dim]
semantic_sample = semantic_feat[0]  # [semantic_dim]

# If audio is [T, D], take mean over T; if [D], use as is
if len(audio_feat.shape) == 2:
    audio_sample = audio_feat.mean(axis=0)
else:
    audio_sample = audio_feat

# === Reshape for model: [B, N, D] ===
visual_sample = torch.tensor(visual_sample, dtype=torch.float32).unsqueeze(0).unsqueeze(1)  # [1, 1, visual_dim]
audio_sample = torch.tensor(audio_sample, dtype=torch.float32).unsqueeze(0).unsqueeze(1)    # [1, 1, audio_dim]
semantic_sample = torch.tensor(semantic_sample, dtype=torch.float32).unsqueeze(0).unsqueeze(1)  # [1, 1, semantic_dim]

# === Model dims (edit if needed) ===
visual_dim = visual_sample.shape[-1]
audio_dim = audio_sample.shape[-1]
semantic_dim = semantic_sample.shape[-1]

model = MultiModalFusion(
    visual_dim=visual_dim,
    audio_dim=audio_dim,
    semantic_dim=semantic_dim,
    fusion_dim=768,
    num_layers=4,
    nhead=8,
    output_dim=2
)

# === Run model ===
output = model(visual_sample, audio_sample, semantic_sample)
print('Model output:', output) 