import torch

def l2_distance(pred, target):
    return torch.norm(pred - target, p=2, dim=-1).mean()
 
def inout_accuracy(pred, target):
    pred_label = (pred > 0).float()
    return (pred_label == target).float().mean() 