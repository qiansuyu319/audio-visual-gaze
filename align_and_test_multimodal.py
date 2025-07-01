import numpy as np
import torch
from models.full_model import MultiModalFusion

def sliding_window_indices(n_frames, window_size, step):
    indices = []
    start = 0
    while start + window_size <= n_frames:
        end = start + window_size
        indices.append((start, end))
        start += step
    return indices

def pool_window(feat, idxs, mode='mean'):
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

visual_feat = np.load('test_data/output_features.npy')
audio_feat = np.load('processed_audio_output/audio_per_frame.npy')
semantic_feat = np.load('semantic_pipeline_output/semantic_per_frame.npy')
gt = np.load('test_data/gaze_label_per_frame.npy')

fps = 25
window_size = 25
step = 1

n_frames = visual_feat.shape[0]
idxs = sliding_window_indices(n_frames, window_size, step)
print(f'Total frames: {n_frames}, total windows: {len(idxs)}')

visual_tokens = pool_window(visual_feat, idxs, mode='mean')
audio_tokens = pool_window(audio_feat, idxs, mode='mean')
semantic_tokens = pool_window(semantic_feat, idxs, mode='mean')
gt_tokens = pool_window(gt, idxs, mode='mean')

print('After pooling/alignment:')
print('visual_tokens:', visual_tokens.shape)
print('audio_tokens:', audio_tokens.shape)
print('semantic_tokens:', semantic_tokens.shape)
print('gt_tokens:', gt_tokens.shape)

visual_tokens = torch.tensor(visual_tokens, dtype=torch.float32).unsqueeze(0)
audio_tokens = torch.tensor(audio_tokens, dtype=torch.float32).unsqueeze(0)
semantic_tokens = torch.tensor(semantic_tokens, dtype=torch.float32).unsqueeze(0)
gt_tokens = torch.tensor(gt_tokens, dtype=torch.float32).unsqueeze(0)

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

output = model(visual_tokens, audio_tokens, semantic_tokens)
print('Model output:', output.shape)