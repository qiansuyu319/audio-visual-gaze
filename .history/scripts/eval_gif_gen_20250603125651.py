import os
import json
import cv2
import torch
import warnings
import imageio
from tqdm import tqdm
from PIL import Image
from gazelle.model import get_gazelle_model
from gazelle.utils import gazefollow_auc, gazefollow_l2

warnings.filterwarnings("ignore", message="The default value of the antialias parameter of all the resizing transforms.*")

@torch.no_grad()
def get_gaze_prediction(image_np, bbox, model, transform, device):
    h, w = image_np.shape[:2]
    img_t = transform(Image.fromarray(image_np)).unsqueeze(0).to(device)
    bbox_norm = [bbox[0]/w, bbox[1]/h, bbox[2]/w, bbox[3]/h]
    preds = model({"images": img_t, "bboxes": [[bbox_norm]]})
    heat = preds['heatmap'][0][0].cpu().numpy()
    pred_y, pred_x = divmod(heat.argmax(), heat.shape[1])
    return int(pred_x * w / heat.shape[1]), int(pred_y * h / heat.shape[0])

def generate_gif_with_gaze_prediction(
    img_root_folder,
    json_root_folder,
    video_id,
    model_name,
    ckpt_path,
    gif_path,
    json_suffix="_GT_VideoFormat.json"
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, transform = get_gazelle_model(model_name)
    model.load_gazelle_state_dict(torch.load(ckpt_path, weights_only=True))
    model.to(device).eval()

    video_folder = os.path.join(img_root_folder, video_id)
    anno_path = os.path.join(json_root_folder, f"{video_id}{json_suffix}")

    with open(anno_path, "r", encoding="utf-8") as f:
        annotations = json.load(f)

    frames = []

    for frame_name, persons in tqdm(annotations.items(), desc=f"Processing {video_id}"):
        img_path = os.path.join(video_folder, frame_name)
        if not os.path.isfile(img_path):
            continue

        img = cv2.imread(img_path)
        if img is None:
            continue

        img_np = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        for p in persons:
            bbox = p.get("bbox")
            if not bbox:
                continue

            gx, gy = get_gaze_prediction(img_np, bbox, model, transform, device)
            x1, y1, x2, y2 = map(int, bbox)

            # draw bounding box
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # draw predicted gaze point
            cv2.circle(img, (gx, gy), 5, (0, 0, 255), -1)

            # draw line from face center to gaze
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            cv2.line(img, (cx, cy), (gx, gy), (255, 0, 0), 2)

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        frames.append(img_rgb)

    if frames:
        imageio.mimsave(gif_path, frames, fps=5)
        print(f"✅ Saved GIF to {gif_path}")
    else:
        print(f"⚠️ No valid frames found for video {video_id}")

# 示例调用方式
if __name__ == "__main__":
    generate_gif_with_gaze_prediction(
        img_root_folder  = r"C:\Users\qians\Desktop\gazelle-main\frames",
        json_root_folder = r"C:\Users\qians\Desktop\VGS-dataset\GT_VideoFormat",
        video_id         = "066",
        model_name       = "gazelle_dinov2_vitl14",
        ckpt_path        = r"C:\Users\qians\Desktop\gazelle-main\gazelle_dinov2_vitl14.pt",
        gif_path         = r"C:\Users\qians\Desktop\gazelle-main\011_gaze_prediction.gif"
    )
