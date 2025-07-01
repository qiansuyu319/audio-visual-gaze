import torch
from PIL import Image
import numpy as np
from gazelle.model import get_gazelle_model
from gazelle.utils import gazefollow_auc, gazefollow_l2

@torch.no_grad()
def evaluate_single_image(image_path, bbox_px, gaze_px, height, width, model_name, ckpt_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on {device}")

    model, transform = get_gazelle_model(model_name)
    model.load_gazelle_state_dict(torch.load(ckpt_path, weights_only=True))
    model.to(device)
    model.eval()

    image = transform(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)

    # Normalize coordinates
    x_min, y_min, x_max, y_max = bbox_px
    gazex, gazey = gaze_px
    bbox_norm = [x_min / width, y_min / height, x_max / width, y_max / height]
    gazex_norm = gazex / width
    gazey_norm = gazey / height

    preds = model.forward({"images": image, "bboxes": [[bbox_norm]]})

    heatmap = preds['heatmap'][0][0]
    auc = gazefollow_auc(heatmap, [gazex_norm], [gazey_norm], height, width)

    avg_l2, min_l2 = gazefollow_l2(heatmap, gazex_norm, gazey_norm)

    print("AUC: {:.4f}".format(auc))
    print("Avg L2: {:.4f}".format(avg_l2))
    print("Min L2: {:.4f}".format(min_l2))

if __name__ == "__main__":
    # 替换为你真实路径
    image_path = "./0021000.jpg"
    bbox_px = (314, 49, 540, 275)
    gaze_px = (428.5, 153.5)
    height = 412800
    width = 640
    model_name = "gazelle_dinov2_vitb14"

    ckpt_path = "C:\\Users\\qians\\Desktop\\gazelle-main\\gazelle_dinov2_vitb14.pt"

    evaluate_single_image(image_path, bbox_px, gaze_px, height, width, model_name, ckpt_path)
