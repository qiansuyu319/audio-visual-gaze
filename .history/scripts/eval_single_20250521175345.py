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
    out_csv="results.csv",
    summary_csv="per_video_metrics.csv"
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 加载模型 & transform
    model, transform = get_gazelle_model(model_name)
    model.load_gazelle_state_dict(torch.load(ckpt_path, weights_only=True))
    model.to(device).eval()

    # 预统计总帧数
    total_frames = 0
    for video_id in sorted(os.listdir(img_root_folder)):
        anno_file = os.path.join(json_root_folder, f"{video_id}{json_suffix}")
        if os.path.isfile(anno_file):
            with open(anno_file, 'r', encoding='utf-8') as f:
                ann = json.load(f)
            total_frames += len(ann)

    # 结果文件初始化
    if os.path.exists(out_csv): os.remove(out_csv)
    if os.path.exists(summary_csv): os.remove(summary_csv)

    # 写入全量记录 CSV header
    pd.DataFrame(columns=["video","frame","person_id","auc","avg_l2","min_l2"]) \
      .to_csv(out_csv, index=False)
    # 写入每视频 summary header
    pd.DataFrame(columns=["video","mean_auc","mean_avg_l2","mean_min_l2"]) \
      .to_csv(summary_csv, index=False)

    overall_bar = tqdm(total=total_frames, desc="Total frames", leave=True)

    # 遍历每个视频子文件夹
    for video_id in sorted(os.listdir(img_root_folder)):
        video_folder = os.path.join(img_root_folder, video_id)
        if not os.path.isdir(video_folder):
            continue

        anno_path = os.path.join(json_root_folder, f"{video_id}{json_suffix}")
        if not os.path.isfile(anno_path):
            print(f"[WARN] Missing annotation file {video_id}{json_suffix}, skipped.")
            continue

        with open(anno_path, "r", encoding="utf-8") as f:
            annotations = json.load(f)

        # 当前视频记录
        video_records = []
        for frame_name, persons in annotations.items():
            img_path = os.path.join(video_folder, frame_name)
            if not os.path.isfile(img_path):
                print(f"[WARN] Missing image {frame_name} in {video_folder}, skipped.")
                overall_bar.update(1)
                continue

            img_np = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)

            for p in persons:
                bbox = p.get("bbox"); gaze = p.get("gaze")
                if bbox is None or gaze is None:
                    continue
                x1,y1,x2,y2 = bbox; gx,gy = gaze
                if x1 <= gx <= x2 and y1 <= gy <= y2:
                    continue

                auc, avg_l2, min_l2 = eval_one_sample(
                    img_np, bbox, gaze, model, transform, device
                )
                rec = {"video": video_id, "frame": frame_name,
                       "person_id": p.get("person_id"),
                       "auc": round(auc,4), "avg_l2": round(avg_l2,4), "min_l2": round(min_l2,4)}
                video_records.append(rec)
                # 追加写入单条记录
                pd.DataFrame([rec]).to_csv(out_csv, mode='a', index=False, header=False)

            overall_bar.update(1)

        # 计算并输出当前视频的平均指标
        if video_records:
            df_vid = pd.DataFrame(video_records)
            m_auc = df_vid["auc"].mean()
            m_avg = df_vid["avg_l2"].mean()
            m_min = df_vid["min_l2"].mean()
            print(f"Video {video_id} → AUC: {m_auc:.4f}, avg_l2: {m_avg:.4f}, min_l2: {m_min:.4f}")
            # 写入 summary CSV
            summary = {"video": video_id,
                       "mean_auc": round(m_auc,4),
                       "mean_avg_l2": round(m_avg,4),
                       "mean_min_l2": round(m_min,4)}
            pd.DataFrame([summary]).to_csv(summary_csv, mode='a', index=False, header=False)

    overall_bar.close()
    print(f"Done! Detailed metrics at {out_csv}, per-video summary at {summary_csv}")

if __name__ == "__main__":
    evaluate_dataset(
        img_root_folder  = r"C:\Users\qians\Desktop\VGS-dataset\JPEGImages",
        json_root_folder = r"C:\Users\qians\Desktop\VGS-dataset\GT_VideoFormat",
        model_name       = "gazelle_dinov2_vitl14",
        ckpt_path        = r"C:\Users\qians\Desktop\gazelle-main\gazelle_dinov2_vitl14.pt",
        json_suffix      = "_GT_VideoFormat.json",
        out_csv          = r"C:\Users\qians\Desktop\gazelle-main\all_gaze_metrics.csv",
        summary_csv      = r"C:\Users\qians\Desktop\gazelle-main\video_summary.csv"
    )
