import os
import json
import warnings
from tqdm import tqdm

import torch
from PIL import Image
import numpy as np
import cv2
import pandas as pd

from gazelle.model import get_gazelle_model
from gazelle.utils import gazefollow_auc, gazefollow_l2

# 屏蔽 antialias 警告
warnings.filterwarnings(
    "ignore",
    message="The default value of the antialias parameter of all the resizing transforms.*"
)

@torch.no_grad()
def eval_one_sample(image_np, bbox, gaze, model, transform, device):
    h, w = image_np.shape[:2]
    img_t = transform(Image.fromarray(image_np)).unsqueeze(0).to(device)
    bbox_norm = [bbox[0]/w, bbox[1]/h, bbox[2]/w, bbox[3]/h]
    gazex_norm, gazey_norm = gaze[0]/w, gaze[1]/h

    preds = model({"images": img_t, "bboxes": [[bbox_norm]]})
    heat = preds['heatmap'][0][0].cpu()

    auc = gazefollow_auc(heat, [gazex_norm], [gazey_norm], h, w)
    avg_l2, min_l2 = gazefollow_l2(heat, gazex_norm, gazey_norm)
    return float(auc), float(avg_l2), float(min_l2)

def evaluate_dataset(
    img_root_folder,
    json_root_folder,
    model_name,
    ckpt_path,
    json_suffix="_GT_VideoFormat.json",
    out_csv="results.csv"
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 加载模型
    model, transform = get_gazelle_model(model_name)
    model.load_gazelle_state_dict(torch.load(ckpt_path, weights_only=True))
    model.to(device).eval()

    all_records = []
    # 遍历每个子文件夹
    for video_id in sorted(os.listdir(img_root_folder)):
        video_folder = os.path.join(img_root_folder, video_id)
        if not os.path.isdir(video_folder):
            continue

        # 构造对应 JSON 文件名：如 011_GT_VideoFormat.json
        anno_filename = f"{video_id}{json_suffix}"
        anno_path = os.path.join(json_root_folder, anno_filename)
        if not os.path.isfile(anno_path):
            print(f"[WARN] Missing annotation {anno_filename}, skipped.")
            continue

        # 读取标注
        with open(anno_path, "r", encoding="utf-8") as f:
            annotations = json.load(f)

        # 遍历帧
        for frame_name, persons in tqdm(annotations.items(), desc=f"Video {video_id}", leave=False):
            img_path = os.path.join(video_folder, frame_name)
            if not os.path.isfile(img_path):
                continue
            img_np = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)

            for p in persons:
                pid = p.get("person_id")
                bbox = p.get("bbox")
                gaze = p.get("gaze")
                if bbox is None or gaze is None:
                    continue
                x1, y1, x2, y2 = bbox
                gx, gy = gaze
                # 跳过框内点
                if x1 <= gx <= x2 and y1 <= gy <= y2:
                    continue

                auc, avg_l2, min_l2 = eval_one_sample(
                    img_np, bbox, gaze, model, transform, device
                )
                all_records.append({
                    "video": video_id,
                    "frame": frame_name,
                    "person_id": pid,
                    "auc": round(auc, 4),
                    "avg_l2": round(avg_l2, 4),
                    "min_l2": round(min_l2, 4),
                })

    # 保存与统计
    df = pd.DataFrame(all_records, columns=["video","frame","person_id","auc","avg_l2","min_l2"])
    df.to_csv(out_csv, index=False)
    print(f"Done! Saved metrics to {out_csv}")

    # 全局平均
    print("Overall averages:")
    print(f"AUC: {df['auc'].mean():.4f}, avg_l2: {df['avg_l2'].mean():.4f}, min_l2: {df['min_l2'].mean():.4f}")

    # 按视频平均
    per_video = df.groupby("video")[['auc','avg_l2','min_l2']].mean().round(4)
    print("Per-video averages:")
    print(per_video)

if __name__ == "__main__":
    evaluate_dataset(
        img_root_folder  = r"C:\Users\qians\Desktop\VGS-dataset\GT_VideoFormat",
        json_root_folder = r"C:\Users\qians\Desktop\VGS-dataset\JPEGImages",
        model_name       = "gazelle_dinov2_vitl14",
        ckpt_path        = r"C:\Users\qians\Desktop\gazelle-main\gazelle_dinov2_vitl14.pt",
        json_suffix      = "_GT_VideoFormat.json",
        out_csv          = r"C:\Users\qians\Desktop\gazelle-main\all_gaze_metrics.csv"
    )
