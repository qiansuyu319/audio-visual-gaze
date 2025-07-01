import numpy as np
import torch
from models.full_model import MultiModalFusion

# === File paths (edit as needed) ===
visual_path = 'test_data/output_features.npy'  # shape: [N, visual_dim] or [B, N, visual_dim]
audio_path = 'processed_audio_output/window_00000_wav2vec.npy'  # shape: [T, audio_dim] or [B, T, audio_dim]
semantic_path = 'semantic_pipeline_output/mclip_embeddings.npy'  # shape: [N, semantic_dim] or [B, N, semantic_dim]

# === Load features ===
visual_feat = np.load(visual_path)  # [N, visual_dim] or [B, N, visual_dim]
audio_feat = np.load(audio_path)    # [T, audio_dim] or [B, T, audio_dim]
semantic_feat = np.load(semantic_path)  # [N, semantic_dim] or [B, N, semantic_dim]

# --- Visual tokens ---
# If shape is [N, D], treat as [1, N, D]
if len(visual_feat.shape) == 2:
    visual_tokens = torch.tensor(visual_feat, dtype=torch.float32).unsqueeze(0)  # [1, N, D]
else:
    visual_tokens = torch.tensor(visual_feat[0], dtype=torch.float32).unsqueeze(0)  # [1, N, D]

# --- Audio tokens ---
if len(audio_feat.shape) == 2:
    audio_tokens = torch.tensor(audio_feat, dtype=torch.float32).unsqueeze(0)  # [1, T, D]
else:
    audio_tokens = torch.tensor(audio_feat[0], dtype=torch.float32).unsqueeze(0)  # [1, T, D]

# --- Semantic tokens ---
if len(semantic_feat.shape) == 2:
    semantic_tokens = torch.tensor(semantic_feat, dtype=torch.float32).unsqueeze(0)  # [1, M, D]
else:
    semantic_tokens = torch.tensor(semantic_feat[0], dtype=torch.float32).unsqueeze(0)  # [1, M, D]

print('visual_tokens shape:', visual_tokens.shape)
print('audio_tokens shape:', audio_tokens.shape)
print('semantic_tokens shape:', semantic_tokens.shape)

# === Model dims (edit if needed) ===
visual_dim = visual_tokens.shape[-1]
audio_dim = audio_tokens.shape[-1]
semantic_dim = semantic_tokens.shape[-1]

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
output = model(visual_tokens, audio_tokens, semantic_tokens)
print('Model output:', output) 