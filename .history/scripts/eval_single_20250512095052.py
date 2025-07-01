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
    heatmap_resized = cv2.resize(heatmap, (width, height))
    pred_y, pred_x = np.unravel_index(np.argmax(heatmap_resized), heatmap_resized.shape)

    print(f"[CHECK] predicted gaze (x, y): ({pred_x}, {pred_y})")
    print(f"[CHECK] ground truth gaze (x_norm, y_norm): ({gazex_norm:.4f}, {gazey_norm:.4f})")

    # === 评估 ===
    auc = gazefollow_auc(torch.tensor(heatmap), [gazex_norm], [gazey_norm], height, width)
    avg_l2, min_l2 = gazefollow_l2(torch.tensor(heatmap), gazex_norm, gazey_norm)
    print(f"AUC: {auc:.4f}\nAvg L2: {avg_l2:.4f}\nMin L2: {min_l2:.4f}")

    # === 可视化 ===
    vis_image = original_image.copy()

    # 绘制头部框
    cv2.rectangle(vis_image, (x_min, y_min), (x_max, y_max), (255, 255, 0), 2)
    cv2.putText(vis_image, "Face", (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    # 注视线（从头部中心到预测点）
    head_center = (int((x_min + x_max) / 2), int((y_min + y_max) / 2))
    cv2.arrowedLine(vis_image, head_center, (pred_x, pred_y), (0, 0, 255), 2, tipLength=0.1)

    # GT gaze 点
    cv2.drawMarker(vis_image, (int(gazex), int(gazey)), (0, 255, 0), cv2.MARKER_CROSS, 15, 2)
    cv2.putText(vis_image, "GT Gaze", (int(gazex) + 5, int(gazey) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Pred gaze 点
    cv2.circle(vis_image, (pred_x, pred_y), 6, (0, 0, 255), -1)
    cv2.putText(vis_image, "Pred Gaze", (pred_x + 5, pred_y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # 添加评估指标文本
    metrics = [
        f"AUC: {auc:.4f}",
        f"Avg L2: {avg_l2:.4f}",
        f"Min L2: {min_l2:.4f}"
    ]
    for i, text in enumerate(metrics):
        cv2.putText(vis_image, text, (10, 30 + i * 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)  # 黑色底层阴影
        cv2.putText(vis_image, text, (10, 30 + i * 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)  # 白色上层字体

    # 显示图像
    plt.figure(figsize=(10, 6))
    plt.imshow(vis_image)
    plt.axis('off')
    plt.title("Gaze Prediction Visualization with Metrics")
    plt.show()

    # 保存图像
    if save_path:
        cv2.imwrite(save_path, cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))
        print(f"Saved visualization to {save_path}")

if __name__ == "__main__":
    image_path = "./.jpg"
    bbox_px = (698,99,960,361)
    gaze_px = (840,205)
    model_name = "gazelle_dinov2_vitl14"
    ckpt_path = r"C:\Users\qians\Desktop\gazelle-main\gazelle_dinov2_vitl14.pt"
    save_output = "./0021000_output.jpg"

    evaluate_single_image(image_path, bbox_px, gaze_px, model_name, ckpt_path, save_path=save_output)
