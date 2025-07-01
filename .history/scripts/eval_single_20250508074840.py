import torch
from PIL import Image
import numpy as np
import argparse

from gazelle.model import get_gazelle_model
from gazelle.utils import gazefollow_auc, gazefollow_l2

@torch.no_grad()
def evaluate_single_image(image_path, bbox_norms, gazex_norms, gazey_norms, height, width, model_name, ckpt_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on {device}")

    model, transform = get_gazelle_model(model_name)
    model.load_gazelle_state_dict(torch.load(ckpt_path, weights_only=True))
    model.to(device)
    model.eval()

    image = transform(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)
    preds = model.forward({"images": image, "bboxes": [bbox_norms]})

    aucs = []
    avg_l2s = []
    min_l2s = []

    for j in range(len(bbox_norms)):
        heatmap = preds['heatmap'][0][j]
        auc = gazefollow_auc(heatmap, gazex_norms[j], gazey_norms[j], height, width)
        avg_l2, min_l2 = gazefollow_l2(heatmap, gazex_norms[j], gazey_norms[j])
        aucs.append(auc)
        avg_l2s.append(avg_l2)
        min_l2s.append(min_l2)

    print("AUC: {:.4f}".format(np.mean(aucs)))
    print("Avg L2: {:.4f}".format(np.mean(avg_l2s)))
    print("Min L2: {:.4f}".format(np.mean(min_l2s)))

if __name__ == "__main__":
    # 示例用法（可以替换为 argparse 或其他输入方式）
    image_path = "./test.jpg"
    bbox_norms = [[0.3, 0.3, 0.4, 0.4]]
    gazex_norms = [0.6]
    gazey_norms = [0.5]
    height = 480
    width = 640
    model_name = "gazelle_dinov2_vitl14"
    ckpt_path = "./checkpoints/gazelle_dinov2_vitl14.pt"

    evaluate_single_image(image_path, bbox_norms, gazex_norms, gazey_norms, height, width, model_name, ckpt_path)
