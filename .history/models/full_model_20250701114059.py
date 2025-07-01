import torch
import torch.nn as nn

class MultiModalFusion(nn.Module):
    def __init__(
        self,
        visual_dim,    # e.g. 768 or 1024
        audio_dim,     # e.g. 768
        semantic_dim,  # e.g. 768
        fusion_dim=768,   # unified token dim
        num_layers=4,
        nhead=8,
        output_dim=2,     # 2 for gaze (x,y), or change for heatmap/classification
    ):
        super().__init__()
        # Project all modalities to unified fusion_dim
        self.visual_proj = nn.Linear(visual_dim, fusion_dim)
        self.audio_proj = nn.Linear(audio_dim, fusion_dim)
        self.semantic_proj = nn.Linear(semantic_dim, fusion_dim)
        # optional: add more proj for other modalities

        # Main transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=fusion_dim, nhead=nhead)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output head: can be MLP for regression/classification
        self.output_head = nn.Linear(fusion_dim, output_dim)

    def forward(self, visual, audio, semantic):
        """
        visual: [B, N1, Vdim] (e.g. patch tokens)
        audio:  [B, N2, Adim] (e.g. utterance tokens, or [B, 1, Adim] for global audio)
        semantic: [B, N3, Sdim] (e.g. [B, 1, Sdim] for global semantic)
        """
        B = visual.size(0)
        # [B, N?, D] -> [B, N?, fusion_dim]
        visual_t = self.visual_proj(visual)
        audio_t = self.audio_proj(audio)
        semantic_t = self.semantic_proj(semantic)
        # Concatenate all tokens: [B, N_total, fusion_dim]
        tokens = torch.cat([visual_t, audio_t, semantic_t], dim=1)
        # Transformer expects [N_total, B, D]
        tokens = tokens.transpose(0, 1)
        fused = self.transformer(tokens)
        # Use the last token for prediction
        out = self.output_head(fused[-1])  # [B, output_dim]
        return out

# Example usage
if __name__ == "__main__":
    B = 4         # batch size
    N1 = 16       # number of visual tokens
    N2 = 2        # number of audio tokens
    N3 = 1        # number of semantic tokens
    visual_dim = 1024
    audio_dim = 768
    semantic_dim = 768

    visual_feat = torch.randn(B, N1, visual_dim)
    audio_feat = torch.randn(B, N2, audio_dim)
    semantic_feat = torch.randn(B, N3, semantic_dim)

    model = MultiModalFusion(
        visual_dim=visual_dim,
        audio_dim=audio_dim,
        semantic_dim=semantic_dim,
        fusion_dim=768,
        num_layers=4,
        nhead=8,
        output_dim=2  
    )

    out = model(visual_feat, audio_feat, semantic_feat)
    print(out.shape)  # [B, 2]
