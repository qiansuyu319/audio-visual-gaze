import torch
import torch.nn as nn

class GazeHead(nn.Module):
    def __init__(self, d_model, heatmap_size):
        super().__init__()
        self.fc = nn.Linear(d_model, heatmap_size * heatmap_size)
        self.heatmap_size = heatmap_size

    def forward(self, x):
        x = self.fc(x)
        return x.view(-1, 1, self.heatmap_size, self.heatmap_size) 