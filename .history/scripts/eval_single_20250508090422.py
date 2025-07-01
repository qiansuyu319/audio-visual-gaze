import torch
from PIL import Image
import numpy as np
from gazelle.model import get_gazelle_model
from gazelle.utils import gazefollow_auc, gazefollow_l2
import matplotlib.pyplot as plt
import cv2

@torch.no_grad()
def evaluate_single_image(image_path, bbox_px, gaze_px, model_name, ckpt_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on {device}")

    # 加载模型
    model, transform = get_gazelle_model(model_name)
    model.load_gazelle_state_dict(torch.load(ckpt_path, weights_only=True))
    model.to(device)
    model.eval()

    # 图像预处理
    image_tensor = transform(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)
    original_image = cv2.imread(image_path)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    height, width = original_image.shape[:2]

    # Normalize bounding box 和 gaze 点
    x_min, y_min, x_max, y_max = bbox_px
    gazex, gazey = gaze_px
    bbox_norm = [x_min / width, y_min / height, x_max / width, y_max / height]
    gazex_norm = gazex / width
    gazey_norm = gazey / height

    # 推理
    preds = model.forward({"images": image_tensor, "bboxes": [[bbox_norm]]})
    heatmap = preds['heatmap'][0][0]

    # 评估指标
    auc = gazefollow_auc(heatmap, [gazex_norm], [gazey_norm], height, width)
    avg_l2, min_l2 = gazefollow_l2(heatmap, gazex_norm, gazey_norm)

    print("AUC: {:.4f}".format(auc))
    print("Avg L2: {:.4f}".format(avg_l2))
    print("Min L2: {:.4f}".format(min_l2))

    # 可视化 heatmap 原图
    heatmap_np = heatmap.cpu().numpy()
    plt.figure(figsize=(6, 5))
    plt.imshow(heatmap_np, cmap='jet')
    plt.title("Raw Heatmap Output")
    plt.colorbar()
    plt.axis('off')
    plt.show()

    # Resize heatmap to original size
    heatmap_resized = cv2.resize(heatmap_np, (width, height))
    pred_y, pred_x = np.unravel_index(np.argmax(heatmap_resized), heatmap_resized.shape)

    # 画框、点
    cv2.rectangle(original_image, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
    cv2.putText(original_image, "Face", (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    gt_x, gt_y = int(gazex), int(gazey)
    cv2.drawMarker(original_image, (gt_x, gt_y), (0, 255, 0), cv2.MARKER_CROSS, 15, 2)
    cv2.putText(original_image, "GT Gaze", (gt_x + 5, gt_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.circle(original_image, (pred_x, pred_y), 8, (0, 0, 255), -1)
    cv2.putText(original_image, "Pred Gaze", (pred_x + 5, pred_y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # 最终图像展示
    plt.figure(figsize=(10, 6))
    plt.imshow(original_image)
    plt.axis('off')
    plt.title("Gaze Prediction Visualization")
    plt.show()

if __name__ == "__main__":
    image_path = "./0021000.jpg"
    bbox_px = (314, 49, 540, 275)
    gaze_px = (745,161)
    model_name = "gazelle_dinov2_vitb14"
    ckpt_path = "C:\\Users\\qians\\Desktop\\gazelle-main\\gazelle_dinov2_vitb14.pt"

    evaluate_single_image(image_path, bbox_px, gaze_px, model_name, ckpt_path)
