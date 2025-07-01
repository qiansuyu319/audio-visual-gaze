import os
import json
import warnings
from tqdm import tqdm

import torch
from PIL import Image
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
    summary_csv="per_video_metrics.csv"
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 加载模型 & transform
    model, transform = get_gazelle_model(model_name)
    model.load_gazelle_state_dict(torch.load(ckpt_path, weights_only=True))
    model.to(device).eval()

    # 统计有效视频数
    video_ids = [d for d in sorted(os.listdir(img_root_folder))
                 if os.path.isdir(os.path.join(img_root_folder, d))
                 and os.path.isfile(os.path.join(json_root_folder, f"{d}{json_suffix}"))]

    # 初始化 summary 文件
    if os.path.exists(summary_csv): os.remove(summary_csv)
    pd.DataFrame(columns=["video","mean_auc","mean_avg_l2","mean_min_l2"]).to_csv(summary_csv, index=False)

    # 整体视频进度条
    overall_bar = tqdm(total=len(video_ids), desc="Videos", leave=True)

    for video_id in video_ids:
        anno_path = os.path.join(json_root_folder, f"{video_id}{json_suffix}")
        with open(anno_path, "r", encoding="utf-8") as f:
            annotations = json.load(f)

        # 视频内部帧进度条
        video_bar = tqdm(total=len(annotations), desc=f"Video {video_id}", leave=False)

        video_records = []
        video_folder = os.path.join(img_root_folder, video_id)

        for frame_name, persons in annotations.items():
            img_path = os.path.join(video_folder, frame_name)
            if not os.path.isfile(img_path):
                video_bar.update(1)
                continue

            img_np = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)

            for p in persons:
                bbox = p.get("bbox"); gaze = p.get("gaze")
                if not bbox or not gaze:
                    continue
                x1,y1,x2,y2 = bbox; gx,gy = gaze
                #if x1 <= gx <= x2 and y1 <= gy <= y2:
                    #continue

                auc, avg_l2, min_l2 = eval_one_sample(
                    img_np, bbox, gaze, model, transform, device
                )
                video_records.append((auc, avg_l2, min_l2))

            video_bar.update(1)
        video_bar.close()

        # 计算并输出当前视频的平均指标
        if video_records:
            arr = pd.DataFrame(video_records, columns=["auc","avg_l2","min_l2"])
            m_auc = arr["auc"].mean()
            m_avg = arr["avg_l2"].mean()
            m_min = arr["min_l2"].mean()
            print(f"Video {video_id} → AUC: {m_auc:.4f}, avg_l2: {m_avg:.4f}, min_l2: {m_min:.4f}")
            # 写入 summary CSV
            summary = {"video": video_id,
                       "mean_auc": round(m_auc,4),
                       "mean_avg_l2": round(m_avg,4),
                       "mean_min_l2": round(m_min,4)}
            pd.DataFrame([summary]).to_csv(summary_csv, mode='a', index=False, header=False)

        overall_bar.update(1)

    overall_bar.close()
    print(f"Done! Per-video summary saved to {summary_csv}")

if __name__ == "__main__":
    evaluate_dataset(
        img_root_folder  = r"C:\Users\qians\Desktop\VGS-dataset\J",
        json_root_folder = r"C:\Users\qians\Desktop\VGS-dataset\GT_VideoFormat",
        model_name       = "gazelle_dinov2_vitl14",
        ckpt_path        = r"C:\Users\qians\Desktop\gazelle-main\gazelle_dinov2_vitl14.pt",
        summary_csv      = r"C:\Users\qians\Desktop\gazelle-main\video_summary_includeAll.csv"
    )