import torch
import torch.nn as nn

def gaze_heatmap_loss(pred, target):
    return nn.functional.mse_loss(pred, target)

def inout_loss(pred, target):
    return nn.functional.binary_cross_entropy_with_logits(pred, target) 