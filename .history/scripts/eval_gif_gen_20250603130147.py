import os
import json
import warnings
import imageio
import numpy as np
import torch
import cv2
from PIL import Image
from tqdm import tqdm
from gazelle.model import get_gazelle_model
from gazelle.utils import gazefollow_auc, gazefollow_l2

warnings.filterwarnings("ignore", message="The default value of the antialias parameter.*")

def draw_visualization(img_np, bbox, gaze_point, pred_point=None):
    img = img_np.copy()
    x1, y1, x2, y2 = map(int, bbox)
    gx, gy = map(int, gaze_point)
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.circle(img, (gx, gy), 5, (0, 0, 255), -1)
    if pred_point:
        px, py = map(int, pred_point)
        cv2.circle(img, (px, py), 5, (255, 255, 0), -1)
        cv2.line(img, ((x1+x2)//2, (y1+y2)//2), (px, py), (255, 255, 0), 2)
    return img

@torch.no_grad()
def generate_gif_from_frames(
    frame_folder,
    json_path,
    model_name,
    ckpt_path,
    output_path="output.gif"
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, transform = get_gazelle_model(model_name)
    model.load_gazelle_state_dict(torch.load(ckpt_path, weights_only=True))
    model.to(device).eval()

    with open(json_path, "r", encoding="utf-8") as f:
        annotations = json.load(f)

    frames = []
    for frame_name, persons in tqdm(annotations.items(), desc="Processing"):
        img_path = os.path.join(frame_folder, frame_name)
        if not os.path.exists(img_path):
            continue

        img_np = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        h, w = img_np.shape[:2]
        img_t = transform(Image.fromarray(img_np)).unsqueeze(0).to(device)

        visual = img_np.copy()
        for p in persons:
            bbox = p.get("bbox")
            gaze = p.get("gaze")
            if not bbox or not gaze:
                continue

            bbox_norm = [bbox[0]/w, bbox[1]/h, bbox[2]/w, bbox[3]/h]
            gaze_norm = [gaze[0]/w, gaze[1]/h]

            preds = model({"images": img_t, "bboxes": [[bbox_norm]]})
            heatmap = preds['heatmap'][0][0].cpu().numpy()
            pred_y, pred_x = np.unravel_index(np.argmax(heatmap), heatmap.shape)
            pred_x = int(pred_x * w / heatmap.shape[1])
            pred_y = int(pred_y * h / heatmap.shape[0])

            visual = draw_visualization(visual, bbox, gaze, (pred_x, pred_y))

        frames.append(visual)

    imageio.mimsave(output_path, frames, duration=0.1)
    print(f"✅ GIF saved at: {output_path}")


if __name__ == "__main__":
    generate_gif_from_frames(
        frame_folder=r"C:\Users\qians\Desktop\gazelle-main\frames",  # 你的帧图像目录
        json_path=r"C:\Users\qians\Desktop\VGS-dataset\GT_VideoFormat\066_GT_VideoFormat.json",  # 对应json
        model_name="gazelle_dinov2_vitl14",
        ckpt_path=r"C:\Users\qians\Desktop\gazelle-main\gazelle_dinov2_vitl14.pt",
        output_path=r"C:\Users\qians\Desktop\gazelle-main\vis_066.gif"
    )
