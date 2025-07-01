import torch
import torch.nn as nn
from .multimodal_transformer import MultimodalTransformer
from .gaze_head import GazeHead

class MultimodalGazelleModel(nn.Module):
    def __init__(self, audio_dim, text_dim, visual_dim, d_model, heatmap_size):
        super().__init__()
        self.transformer = MultimodalTransformer(audio_dim, text_dim, visual_dim, d_model)
        self.gaze_head = GazeHead(d_model, heatmap_size)
        self.cls_token = nn.Linear(d_model, 1)  # For in/out prediction

    def forward(self, audio_feat, text_feat, visual_feat):
        fused = self.transformer(audio_feat, text_feat, visual_feat)
        gaze_heatmap = self.gaze_head(fused[:, 0])
        inout_pred = self.cls_token(fused[:, 0])
        return gaze_heatmap, inout_pred 