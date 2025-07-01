import os
import json
import torch
from PIL import Image
import numpy as np
import cv2
import pandas as pd
from gazelle.model import get_gazelle_model
from gazelle.utils import gazefollow_auc, gazefollow_l2

@torch.no_grad()
def eval_one_sample(image_np, bbox, gaze, model, transform, device):
    """
    对单个人脸样本做推理并返回指标。
    image_np: H×W×3 RGB numpy
    bbox: [x_min,y_min,x_max,y_max]
    gaze: [gazex,gazey]
    """
    h, w = image_np.shape[:2]
    # 前处理 & forward
    img_t = transform(Image.fromarray(image_np)).unsqueeze(0).to(device)
    bbox_norm = [bbox[0]/w, bbox[1]/h, bbox[2]/w, bbox[3]/h]
    gazex_norm, gazey_norm = gaze[0]/w, gaze[1]/h

    preds = model({"images": img_t, "bboxes": [[bbox_norm]]})
    heat = preds['heatmap'][0][0].cpu()

    # 评估指标（gazefollow_auc 和 gazefollow_l2 都可能直接返回 float）
    auc = gazefollow_auc(heat, [gazex_norm], [gazey_norm], h, w)
    avg_l2, min_l2 = gazefollow_l2(heat, gazex_norm, gazey_norm)

    # 确保都是 float
    return float(auc), float(avg_l2), float(min_l2)

def evaluate_video_folder(
    img_folder,          # 存视频帧的文件夹
    anno_json_path,      # 上述整段视频的 JSON 标注文件
    model_name, ckpt_path,
    proximity_thresh_px=20,
    out_csv="results.csv"
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device}")
    # 加载模型
    model, transform = get_gazelle_model(model_name)
    model.load_gazelle_state_dict(torch.load(ckpt_path, weights_only=True))
    model.to(device).eval()

    # 载入所有标注
    with open(anno_json_path, "r") as f:
        annotations = json.load(f)

    records = []
    # 遍历每一帧、每个人
    for frame_name, persons in annotations.items():
        img_path = os.path.join(img_folder, frame_name)
        if not os.path.isfile(img_path):
            print(f"[WARN] Missing frame image {frame_name}, skipped.")
            continue
        img_np = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)

        for p in persons:
            pid   = p["person_id"]
            bbox  = p["bbox"]
            gaze  = p["gaze"]

            # 跳过过近样本
            x1,y1,x2,y2 = bbox
            gx,gy       = gaze
            if (x1 - proximity_thresh_px) <= gx <= (x2 + proximity_thresh_px) and \
               (y1 - proximity_thresh_px) <= gy <= (y2 + proximity_thresh_px):
                continue

            # 计算指标
            auc, avg_l2, min_l2 = eval_one_sample(img_np, bbox, gaze,
                                                  model, transform, device)

            records.append({
                "frame":      frame_name,
                "person_id":  pid,
                "auc":        round(auc, 4),
                "avg_l2":     round(avg_l2, 4),
                "min_l2":     round(min_l2, 4)
            })

    # 保存到 CSV
    df = pd.DataFrame.from_records(records,
        columns=["frame","person_id","auc","avg_l2","min_l2"])
    df.to_csv(out_csv, index=False)
    print(f"Done! Saved metrics to {out_csv}")

if __name__ == "__main__":
    evaluate_video_folder(
        img_folder         = "C:\Users\qians\Desktop\gazelle-main\002",
        anno_json_path     = r"C:\Users\qians\Desktop\gazelle-main\002_GT_VideoFormat.json",
        model_name         = "gazelle_dinov2_vitl14",
        ckpt_path          = r"C:\Users\qians\Desktop\gazelle-main\gazelle_dinov2_vitl14.pt",
        proximity_thresh_px=20,
        out_csv            = "./gaze_metrics.csv"
    )
