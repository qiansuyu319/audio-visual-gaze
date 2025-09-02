from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from transformers import AutoModel
import torchvision.transforms as transforms

# Abstract Backbone class
class Backbone(nn.Module, ABC):
    def __init__(self):
        super(Backbone, self).__init__()
    
    @abstractmethod
    def forward(self, x):
        pass

    @abstractmethod
    def get_dimension(self):
        pass

    @abstractmethod
    def get_out_size(self, in_size):
        pass

    def get_transform(self):
        pass


# Official DINOv2 backbones from torch hub (https://github.com/facebookresearch/dinov2#pretrained-backbones-via-pytorch-hub)
class DinoV2Backbone(Backbone):
    def __init__(self, model_name):
        super(DinoV2Backbone, self).__init__()
        self.model = AutoModel.from_pretrained("facebook/dinov2-base")

    def forward(self, x):
        b, c, h, w = x.shape
        out_h, out_w = self.get_out_size((h, w))
        outputs = self.model(x)
        x = outputs.last_hidden_state[:, 1:]  # Remove CLS token, keep patch tokens
        x = x.view(x.size(0), out_h, out_w, -1).permute(0, 3, 1, 2) # "b (out_h out_w) c -> b c out_h out_w"
        return x
    
    def get_dimension(self):
        return self.model.config.hidden_size
    
    def get_out_size(self, in_size):
        h, w = in_size
        return (h // self.model.config.patch_size, w // self.model.config.patch_size)
    
    def get_transform(self, in_size):
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485,0.456,0.406],
                std=[0.229,0.224,0.225]
            ),
            transforms.Resize(in_size),
        ])