import numpy as np
import torch
from models.full_model import MultiModalFusion

def sliding_window_indices(n_frames, window_size, step):
    """Generate all sliding window (start, end) index pairs."""
    indices = []
    start = 0
    while start + window_size <= n_frames:
        end = start + window_size
        indices.append((start, end))
        start += step
    return indices

def pool_window(feat, idxs, mode='mean'):
    """Pool feature [N_frames, D] to [N_windows, D] with given indices."""
    pooled = []
    for s, e in idxs:
        chunk = feat[s:e]
        if len(chunk) == 0:
            pooled.append(np.zeros(feat.shape[1]))
        elif mode == 'mean':
            pooled.append(chunk.mean(axis=0))
        elif mode == 'last':
            pooled.append(chunk[-1])
    return np.stack(pooled)

# === Load per-frame features & label (high freq, e.g. 25fps) ===
visual_feat = np.load('test_data/output_features.npy')    # [N_frames, visual_dim]
audio_feat = np.load('processed_audio_output/audio_per_frame.npy')  # [N_frames, audio_dim]
semantic_feat = np.load('semantic_pipeline_output/semantic_per_frame.npy')  # [N_frames, semantic_dim]
gt = np.load('test_data/gaze_label_per_frame.npy')        # [N_frames, 2]

# === Define sliding window params ===
fps = 25
window_size = 25  # 1.0 second window (25 frames)
step = 1          # 0.04 second stride (1 frame)

n_frames = visual_feat.shape[0]
idxs = sliding_window_indices(n_frames, window_size, step)
print(f'Total frames: {n_frames}, total windows: {len(idxs)}')

# === Pool all modalities and gt to windows ===
visual_tokens = pool_window(visual_feat, idxs, mode='mean')      # [N_windows, visual_dim]
audio_tokens = pool_window(audio_feat, idxs, mode='mean')        # [N_windows, audio_dim]
semantic_tokens = pool_window(semantic_feat, idxs, mode='mean')  # [N_windows, semantic_dim]
gt_tokens = pool_window(gt, idxs, mode='mean')                   # [N_windows, 2]

print('After pooling/alignment:')
print('visual_tokens:', visual_tokens.shape)
print('audio_tokens:', audio_tokens.shape)
print('semantic_tokens:', semantic_tokens.shape)
print('gt_tokens:', gt_tokens.shape)

# === Convert to model input shape [B, N, D] ===
visual_tokens = torch.tensor(visual_tokens, dtype=torch.float32).unsqueeze(0)      # [1, N, visual_dim]
audio_tokens = torch.tensor(audio_tokens, dtype=torch.float32).unsqueeze(0)        # [1, N, audio_dim]
semantic_tokens = torch.tensor(semantic_tokens, dtype=torch.float32).unsqueeze(0)  # [1, N, semantic_dim]
gt_tokens = torch.tensor(gt_tokens, dtype=torch.float32).unsqueeze(0)              # [1, N, 2]

# === Model dims and create fusion model ===
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
print('Model output:', output.shape)

# === Compute loss (example: MSE) ===
criterion = torch.nn.MSELoss()
loss = criterion(output, gt_tokens)
print('MSE loss:', loss.item())
