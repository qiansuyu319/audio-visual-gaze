import torch
from PIL import Image
import numpy as np
from gazelle.model import get_gazelle_model
from gazelle.utils import gazefollow_auc, gazefollow_l2
import matplotlib.pyplot as plt
import cv2
import os

@torch.no_grad()
def evaluate_single_image(image_path, bbox_px, gaze_px, model_name, ckpt_path, save_path=None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on {device}")

    # === 加载模型 ===
    model, transform = get_gazelle_model(model_name)
    model.load_gazelle_state_dict(torch.load(ckpt_path, weights_only=True))
    model.to(device)
    model.eval()

    # === 图像预处理 ===
    image_tensor = transform(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)
    original_image = cv2.imread(image_path)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    height, width = original_image.shape[:2]

    # === Normalize ===
    x_min, y_min, x_max, y_max = bbox_px
    gazex, gazey = gaze_px
    bbox_norm = [x_min / width, y_min / height, x_max / width, y_max / height]
    gazex_norm = gazex / width
    gazey_norm = gazey / height

    # === 推理 ===
    preds = model.forward({"images": image_tensor, "bboxes": [[bbox_norm]]})
    heatmap = preds['heatmap'][0][0].cpu().numpy()

    print(f"[CHECK] heatmap max: {np.max(heatmap):.6f}, sum: {np.sum(heatmap):.6f}, shape: {heatmap.shape}")
    heatmap_resized = cv2.resize(heatmap, (width, height))
    pred_y, pred_x = np.unravel_index(np.argmax(heatmap_resized), heatmap_resized.shape)
    print(f"[CHECK] predicted gaze (x, y): ({pred_x}, {pred_y})")
    print(f"[CHECK] ground truth gaze (x_norm, y_norm): ({gazex_norm:.4f}, {gazey_norm:.4f})")

    # === 评估 ===
    auc = gazefollow_auc(torch.tensor(heatmap), [gazex_norm], [gazey_norm], height, width)
    avg_l2, min_l2 = gazefollow_l2(torch.tensor(heatmap), gazex_norm, gazey_norm)
    print(f"AUC: {auc:.4f}\nAvg L2: {avg_l2:.4f}\nMin L2: {min_l2:.4f}")

    # === 可视化 heatmap ===
    plt.figure(figsize=(10, 6))

    # 显示原图作为背景
    plt.imshow(original_image, alpha=0.6)
    # 叠加热力图
    plt.imshow(heatmap_resized, cmap='jet', alpha=0.4)

    # 标出 GT 和 Pred gaze
    plt.scatter([gazex], [gazey], color='lime', s=100, marker='x', label='GT Gaze')
    plt.scatter([pred_x], [pred_y], color='red', s=100, marker='o', label='Pred Gaze')

    plt.title("Gaze Prediction Heatmap")
    plt.legend(loc='upper right')
    plt.axis('off')
    plt.tight_layout()
    plt.show()

    # === 保存融合图像 ===
    if save_path:
        heatmap_color = cv2.applyColorMap((heatmap_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
        blend = cv2.addWeighted(original_image, 0.6, heatmap_color, 0.4, 0)
        # 标注点
        cv2.drawMarker(blend, (int(gazex), int(gazey)), (0, 255, 0), cv2.MARKER_CROSS, 15, 2)
        cv2.circle(blend, (pred_x, pred_y), 8, (0, 0, 255), -1)
        cv2.imwrite(save_path, cv2.cvtColor(blend, cv2.COLOR_RGB2BGR))
        print(f"Saved heatmap visualization to {save_path}")

if __name__ == "__main__":
    image_path = "./0021046.jpg"
    bbox_px = (310, 52, 536, 278)
    gaze_px = (797, 177)
    model_name = "gazelle_dinov2_vitl14"
    ckpt_path = r"C:\Users\qians\Desktop\gazelle-main\gazelle_dinov2_vitl14.pt"
    save_output = "./0021000_output.jpg"

    evaluate_single_image(image_path, bbox_px, gaze_px, model_name, ckpt_path, save_path=save_output)
