import torch
from PIL import Image
import numpy as np
from gazelle.model import get_gazelle_model
from gazelle.utils import gazefollow_auc, gazefollow_l2

@torch.no_grad()
def main():
    # === 模型设置 ===
    model_name = "gazelle_dinov2_vitb14"
    ckpt_path = r"C:\Users\qians\Desktop\gazelle-main\checkpoints\gazelle_dinov2_vitb14.pt"

    # === 图片与注视数据 ===
    image_path = r"C:\Users\qians\Desktop\gazelle-main\0021000.jpg"
    bbox_px = (314, 49, 540, 275)
    gaze_px = (441, 154)
    image_height = 480
    image_width = 640

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on {device}")

    # === 加载模型 ===
    model, transform = get_gazelle_model(model_name)
    model.load_gazelle_state_dict(torch.load(ckpt_path, weights_only=True))
    model.to(device)
    model.eval()

    # === 处理图像 ===
    image = transform(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)

    # === 归一化 bbox 和 gaze ===
    x_min, y_min, x_max, y_max = bbox_px
    gazex, gazey = gaze_px
    bbox_norm = [x_min / image_width, y_min / image_height, x_max / image_width, y_max / image_height]
    gazex_norm = gazex / image_width
    gazey_norm = gazey / image_height

    # === 推理并评估 ===
    preds = model.forward({"images": image, "bboxes": [[bbox_norm]]})
    heatmap = preds['heatmap'][0][0]

    auc = gazefollow_auc(heatmap, [gazex_norm], [gazey_norm], image_height, image_width)
    avg_l2, min_l2 = gazefollow_l2(heatmap, gazex_norm, gazey_norm)

    print("AUC: {:.4f}".format(auc))
    print("Avg L2: {:.4f}".format(avg_l2))
    print("Min L2: {:.4f}".format(min_l2))

if __name__ == "__main__":
    main()
