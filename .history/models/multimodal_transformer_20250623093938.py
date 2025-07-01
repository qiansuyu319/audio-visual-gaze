import torch
import torch.nn as nn

class MultimodalTransformer(nn.Module):
    def __init__(self, audio_dim, text_dim, visual_dim, d_model):
        super().__init__()
        self.audio_proj = nn.Linear(audio_dim, d_model)
        self.text_proj = nn.Linear(text_dim, d_model)
        self.visual_proj = nn.Linear(visual_dim, d_model)
        self.transformer = nn.Transformer(d_model=d_model, batch_first=True)

    def forward(self, audio_feat, text_feat, visual_feat):
        a = self.audio_proj(audio_feat)
        t = self.text_proj(text_feat)
        v = self.visual_proj(visual_feat)
        fused = torch.stack([a, t, v], dim=1)
        out = self.transformer(fused, fused)
        return out 