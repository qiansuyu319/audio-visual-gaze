import torch
import torch.nn as nn
import torchvision
from timm.models.vision_transformer import Block
import math

import gazelle.utils as utils
from gazelle.backbone import DinoV2Backbone

import torch
import torch.nn as nn
import torchvision
from timm.models.vision_transformer import Block
import math

import gazelle.utils as utils
from gazelle.backbone import DinoV2Backbone


class GazeLLE(nn.Module):
    def __init__(
        self,
        backbone,
        inout: bool = False,
        dim: int = 256,
        num_layers: int = 3,
        in_size=(448, 448),
        out_size=(64, 64),
        use_audio: bool = False,
        audio_dim: int = 856,
        fusion_dropout: float = 0.1,
    ):
        super().__init__()
        self.backbone = backbone
        self.dim = dim
        self.num_layers = num_layers
        self.featmap_h, self.featmap_w = backbone.get_out_size(in_size)
        self.in_size = in_size
        self.out_size = out_size
        self.inout = inout

        self.linear = nn.Conv2d(backbone.get_dimension(), self.dim, 1)
        self.head_token = nn.Embedding(1, self.dim)
        self.register_buffer(
            "pos_embed",
            positionalencoding2d(self.dim, self.featmap_h, self.featmap_w)
            .squeeze(dim=0)
            .squeeze(dim=0),
        )
        if self.inout:
            self.inout_token = nn.Embedding(1, self.dim)

        self.transformer = nn.Sequential(
            *[
                Block(dim=self.dim, num_heads=8, mlp_ratio=4, drop_path=0.1)
                for _ in range(num_layers)
            ]
        )

        self.heatmap_head = nn.Sequential(
            nn.ConvTranspose2d(dim, dim, kernel_size=2, stride=2),
            nn.Conv2d(dim, 1, kernel_size=1, bias=False),
            nn.Sigmoid(),
        )
        if self.inout:
            self.inout_head = nn.Sequential(
                nn.Linear(self.dim, 128),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(128, 1),
                nn.Sigmoid(),
            )

        self.use_audio = use_audio
        self.audio_dim = audio_dim
        if self.use_audio:
            self.audio_alpha = nn.Parameter(torch.full((1,), 0.1))
            self.audio_norm = nn.LayerNorm(self.audio_dim)
            self.audio_fc = nn.Linear(self.audio_dim, self.dim)
            self.audio_drop = nn.Dropout(p=fusion_dropout)

    def _repeat_audio_per_head(self, audio_feats: torch.Tensor, num_ppl_per_img):
        """
        将 (B, A) 的帧级音频复制到每个 head -> (B*H, A)
        """
        chunks = []
        for i, h in enumerate(num_ppl_per_img):
            if h > 0:
                chunks.append(audio_feats[i].unsqueeze(0).repeat(h, 1))
        if len(chunks) == 0:
            return audio_feats.new_zeros((0, audio_feats.size(-1)))
        return torch.cat(chunks, dim=0)

    def forward(self, input):
        # input["images"]: [B, 3, H, W]
        # input["bboxes"]: list[list[(xmin, ymin, xmax, ymax)]] 归一化坐标
        # 可选：input["audio"]: (B, A) 或 (B*H, A)
        num_ppl_per_img = [len(bbox_list) for bbox_list in input["bboxes"]]

        # --- 视觉主干到空间特征 ---
        x = self.backbone.forward(input["images"])           # (B, Cb, Hf, Wf)
        x = self.linear(x)                                   # (B, dim, Hf, Wf)
        x = x + self.pos_embed                               # 加位置编码
        x = utils.repeat_tensors(x, num_ppl_per_img)         # (B*H, dim, Hf, Wf)

        # --- head map embedding（原逻辑不变） ---
        head_maps = torch.cat(self.get_input_head_maps(input["bboxes"]), dim=0).to(x.device)  # (B*H, Hf, Wf)
        head_map_embeddings = head_maps.unsqueeze(dim=1) * self.head_token.weight.unsqueeze(-1).unsqueeze(-1)  # (B*H, dim, Hf, Wf)
        x = x + head_map_embeddings

        # --- 新增：最小残差音频融合（在 flatten 之前，以通道偏置形式注入；对所有空间 token 生效） ---
        if self.use_audio and ("audio" in input) and (input["audio"] is not None):
            a = input["audio"]
            if a.dim() == 2:  # (B, A) -> (B*H, A)
                a = self._repeat_audio_per_head(a.to(x.device), num_ppl_per_img)
            else:
                a = a.to(x.device)  # 认为已是 (B*H, A)
            if a.numel() > 0:  # 以防空 batch
                a = self.audio_drop(self.audio_fc(self.audio_norm(a)))  # (B*H, dim)
                # 作为通道偏置加到每个空间位置（广播到 Hf×Wf）
                x = x + self.audio_alpha * a.unsqueeze(-1).unsqueeze(-1)

        # --- 进入 Transformer（原逻辑不变） ---
        x = x.flatten(start_dim=2).permute(0, 2, 1)  # (B*H, Hf*Wf, dim)
        if self.inout:
            x = torch.cat(
                [self.inout_token.weight.unsqueeze(dim=0).repeat(x.shape[0], 1, 1), x],
                dim=1,
            )
        x = self.transformer(x)

        # --- in/out 支路（原逻辑不变） ---
        if self.inout:
            inout_tokens = x[:, 0, :]
            inout_preds = self.inout_head(inout_tokens).squeeze(dim=-1)
            inout_preds = utils.split_tensors(inout_preds, num_ppl_per_img)
            x = x[:, 1:, :]  
        
        x = x.reshape(x.shape[0], self.featmap_h, self.featmap_w, x.shape[2]).permute(0, 3, 1, 2)  # (B*H, dim, Hf, Wf)
        x = self.heatmap_head(x).squeeze(dim=1)  # (B*H, Hf*2, Wf*2) after deconv
        x = torchvision.transforms.functional.resize(x, self.out_size)
        heatmap_preds = utils.split_tensors(x, num_ppl_per_img)

        return {"heatmap": heatmap_preds, "inout": inout_preds if self.inout else None}

    def get_input_head_maps(self, bboxes):
        # bboxes: [[(xmin, ymin, xmax, ymax)]] - list of list of head bboxes per image
        head_maps = []
        for bbox_list in bboxes:
            img_head_maps = []
            for bbox in bbox_list:
                if bbox is None: # no bbox provided, use empty head map
                    img_head_maps.append(torch.zeros(self.featmap_h, self.featmap_w))
                else:
                    xmin, ymin, xmax, ymax = bbox
                    width, height = self.featmap_w, self.featmap_h
                    xmin = round(xmin * width)
                    ymin = round(ymin * height)
                    xmax = round(xmax * width)
                    ymax = round(ymax * height)
                    head_map = torch.zeros((height, width))
                    head_map[ymin:ymax, xmin:xmax] = 1
                    img_head_maps.append(head_map)
            head_maps.append(torch.stack(img_head_maps))
        return head_maps

    # ===== Checkpoint utilities (save/load only decoder weights, not backbone) =====
    def get_gazelle_state_dict(self):
        """
        Returns a state_dict that excludes the backbone weights. Intended for
        saving lightweight checkpoints containing only the gaze decoder.
        """
        full_state = self.state_dict()
        decoder_state = {k: v for k, v in full_state.items() if not k.startswith("backbone.")}
        return decoder_state

    def load_gazelle_state_dict(self, checkpoint_like, strict: bool = False):
        """
        Loads a checkpoint that contains only the decoder weights. Handles
        common wrapping formats and prefixes (e.g., 'state_dict', 'model',
        and 'module.' prefix from DDP). Ignores any backbone.* keys.

        Args:
            checkpoint_like: A state_dict or a dict containing a state dict under
                              'state_dict' or 'model'.
            strict: If True, enforce that the keys match exactly (excluding backbone).

        Returns:
            The result of nn.Module.load_state_dict for visibility.
        """
        # Unwrap checkpoint containers
        if not isinstance(checkpoint_like, dict):
            raise TypeError("checkpoint_like must be a dict or state_dict mapping")

        if "state_dict" in checkpoint_like and isinstance(checkpoint_like["state_dict"], dict):
            state_dict = checkpoint_like["state_dict"]
        elif "model" in checkpoint_like and isinstance(checkpoint_like["model"], dict):
            state_dict = checkpoint_like["model"]
        else:
            # Assume this is already a raw state_dict mapping
            state_dict = checkpoint_like

        # Strip common top-level prefixes
        def strip_prefix(key: str):
            prefixes = ["module.", "model.", "net.", "network."]
            for p in prefixes:
                if key.startswith(p):
                    return key[len(p):]
            return key

        state_dict = {strip_prefix(k): v for k, v in state_dict.items()}

        # Filter out backbone weights from the incoming state dict
        state_dict = {k: v for k, v in state_dict.items() if not k.startswith("backbone.")}

        own_state = self.state_dict()

        # Keep only keys that exist in the current model (and are not backbone)
        filtered = {k: v for k, v in state_dict.items() if (k in own_state) and (not k.startswith("backbone."))}

        # Load non-strictly by default to tolerate minor diffs (e.g., heads)
        load_result = super().load_state_dict(filtered, strict=False)

        # Optionally enforce strictness on decoder-side keys only
        if strict:
            # Compute missing/unexpected considering only non-backbone keys
            non_backbone_keys = {k for k in own_state.keys() if not k.startswith("backbone.")}
            missing = sorted(list(non_backbone_keys.difference(filtered.keys())))
            unexpected = sorted([k for k in state_dict.keys() if (k not in own_state) and (not k.startswith("backbone."))])
            if len(missing) > 0 or len(unexpected) > 0:
                raise RuntimeError(
                    f"Error(s) in loading state_dict for GazeLLE (decoder only):\n"
                    f"Missing keys (decoder): {missing}\n"
                    f"Unexpected keys (decoder): {unexpected}"
                )

        return load_result
def positionalencoding2d(d_model, height, width):
    """
    :param d_model: dimension of the model
    :param height: height of the positions
    :param width: width of the positions
    :return: d_model*height*width position matrix
    """
    if d_model % 4 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dimension (got dim={:d})".format(d_model))
    pe = torch.zeros(d_model, height, width)
    # Each dimension use half of d_model
    d_model = int(d_model / 2)
    div_term = torch.exp(torch.arange(0., d_model, 2) *
                         -(math.log(10000.0) / d_model))
    pos_w = torch.arange(0., width).unsqueeze(1)
    pos_h = torch.arange(0., height).unsqueeze(1)
    pe[0:d_model:2, :, :] = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[1:d_model:2, :, :] = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[d_model::2, :, :] = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
    pe[d_model + 1::2, :, :] = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)

    return pe
    

# models
def get_gazelle_model(model_name, use_audio: bool = False, audio_dim: int = 856, fusion_dropout: float = 0.1):
    factory = {
        "gazelle_dinov2_vitb14": lambda: gazelle_dinov2_vitb14(use_audio, audio_dim, fusion_dropout),
        "gazelle_dinov2_vitl14": lambda: gazelle_dinov2_vitl14(use_audio, audio_dim, fusion_dropout),
        "gazelle_dinov2_vitb14_inout": lambda: gazelle_dinov2_vitb14_inout(use_audio, audio_dim, fusion_dropout),
        "gazelle_dinov2_vitl14_inout": lambda: gazelle_dinov2_vitl14_inout(use_audio, audio_dim, fusion_dropout),
    }
    assert model_name in factory.keys(), "invalid model name"
    return factory[model_name]()

def gazelle_dinov2_vitb14(use_audio=False, audio_dim=856, fusion_dropout=0.1):
    backbone = DinoV2Backbone('dinov2_vitb14')
    transform = backbone.get_transform((448, 448))
    model = GazeLLE(backbone, use_audio=use_audio, audio_dim=audio_dim, fusion_dropout=fusion_dropout)
    return model, transform

def gazelle_dinov2_vitl14(use_audio=False, audio_dim=856, fusion_dropout=0.1):
    backbone = DinoV2Backbone('dinov2_vitl14')
    transform = backbone.get_transform((448, 448))
    model = GazeLLE(backbone, use_audio=use_audio, audio_dim=audio_dim, fusion_dropout=fusion_dropout)
    return model, transform

def gazelle_dinov2_vitb14_inout(use_audio=False, audio_dim=856, fusion_dropout=0.1):
    backbone = DinoV2Backbone('dinov2_vitb14')
    transform = backbone.get_transform((448, 448))
    model = GazeLLE(backbone, inout=True, use_audio=use_audio, audio_dim=audio_dim, fusion_dropout=fusion_dropout)
    return model, transform

def gazelle_dinov2_vitl14_inout(use_audio=False, audio_dim=856, fusion_dropout=0.1):
    backbone = DinoV2Backbone('dinov2_vitl14')
    transform = backbone.get_transform((448, 448))
    model = GazeLLE(backbone, inout=True, use_audio=use_audio, audio_dim=audio_dim, fusion_dropout=fusion_dropout)
    return model, transform
