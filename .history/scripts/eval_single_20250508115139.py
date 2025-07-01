import torch
from PIL import Image
import numpy as np
from gazelle.model import get_gazelle_model
from gazelle.utils import gazefollow_l2
import matplotlib.pyplot as plt
import cv2
import os
from sklearn.metrics import roc_auc_score

def bbox_to_gaze_heatmap(x1, y1, x2, y2, image_w, image_h, heatmap_size=(64, 64), sigma=0.03):
    """
    Convert bounding box to Gaussian heatmap centered at its center (normalized).
    """
    H, W = heatmap_size
    cx = (x1 + x2) / 2 / image_w
    cy = (y1 + y2) / 2 / image_h

    xs = np.linspace(0, 1, W)
    ys = np.linspace(0, 1, H)
    x_grid, y_grid = np.meshgrid(xs, ys)

    heatmap = np.exp(-((x_grid - cx)**2 + (y_grid - cy)**2) / (2 * sigma**2))
    heatmap /= np.max(heatmap)
    return heatmap

@torch.no_grad()
def evaluate_single_image(image_path, bbox_px, gaze_target_bbox, model_name, ckpt_path, save_path=None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on {device}")

    # === Load model ===
    model, transform = get_gazelle_model(model_name)
    model.load_gazelle_state_dict(torch.load(ckpt_path, weights_only=True))
    model.to(device)
    model.eval()

    # === Load and preprocess image ===
    image_tensor = transform(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)
    original_image = cv2.imread(image_path)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    height, width = original_image.shape[:2]

    # === Normalize face bounding box ===
    x_min, y_min, x_max, y_max = bbox_px
    bbox_norm = [x_min / width, y_min / height, x_max / width, y_max / height]

    # === Inference ===
    preds = model.forward({"images": image_tensor, "bboxes": [[bbox_norm]]})
    heatmap = preds['heatmap'][0][0].cpu().numpy()

    # === Create GT heatmap from gaze target bounding box ===
    gx1, gy1, gx2, gy2 = gaze_target_bbox
    gt_heatmap = bbox_to_gaze_heatmap(gx1, gy1, gx2, gy2, width, height, heatmap_size=heatmap.shape)

    # === Evaluate ===
    auc = roc_auc_score(gt_heatmap.flatten(), heatmap.flatten())
    
    # Get GT center as point for L2
    gt_cx = (gx1 + gx2) / 2 / width
    gt_cy = (gy1 + gy2) / 2 / height
    avg_l2, min_l2 = gazefollow_l2(torch.tensor(heatmap), gt_cx, gt_cy)
    print(f"AUC: {auc:.4f}\nAvg L2: {avg_l2:.4f}\nMin L2: {min_l2:.4f}")

    # === Predicted gaze point ===
    heatmap_resized = cv2.resize(heatmap, (width, height))
    pred_y, pred_x = np.unravel_index(np.argmax(heatmap_resized), heatmap_resized.shape)

    # === Visualization ===
    cv2.rectangle(original_image, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
    cv2.putText(original_image, "Face", (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    cv2.rectangle(original_image, (gx1, gy1), (gx2, gy2), (255, 165, 0), 2)
    cv2.putText(original_image, "Gaze Target", (gx1, gy1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 165, 0), 2)

    cv2.circle(original_image, (pred_x, pred_y), 8, (0, 0, 255), -1)
    cv2.putText(original_image, "Pred Gaze", (pred_x+5, pred_y+5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # === Overlay metrics ===
    metrics = [
        f"AUC: {auc:.4f}",
        f"Avg L2: {avg_l2:.4f}",
        f"Min L2: {min_l2:.4f}"
    ]
    for i, text in enumerate(metrics):
        cv2.putText(original_image, text, (10, 30 + i * 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
        cv2.putText(original_image, text, (10, 30 + i * 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

    # === Show and Save ===
    plt.figure(figsize=(10, 6))
    plt.imshow(original_image)
    plt.axis('off')
    plt.title("Gaze Prediction Visualization")
    plt.show()

    if save_path:
        cv2.imwrite(save_path, cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR))
        print(f"Saved visualization to {save_path}")

# === Example usage ===
if __name__ == "__main__":
    image_path = "./0021000.jpg"
    bbox_px = (314, 49, 540, 275)
    gaze_target_bbox = (395, 135, 487, 174)
    model_name = "gazelle_dinov2_vitl14"
    ckpt_path = r"C:\Users\qians\Desktop\gazelle-main\gazelle_dinov2_vitl14.pt"
    save_output = "./0021000_output.jpg"

    evaluate_single_image(image_path, bbox_px, gaze_target_bbox, model_name, ckpt_path, save_path=save_output)
