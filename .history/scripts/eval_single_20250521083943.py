import os
import json
import torch
from PIL import Image
import numpy as np
import cv2
import pandas as pd
from gazelle.model import get_gazelle_model
from gazelle.utils import gazefollow_auc, gazefollow_l2
import warnings
from tqdm import tqdm

# 1. 屏蔽 antialias 警告
warnings.filterwarnings(
    "ignore",
    message="The default value of the antialias parameter of all the resizing transforms.*"
)

# 如果你能修改 transform 定义的话，更推荐这样做（以 Resize 为例）：
# from torchvision import transforms
# transform = transforms.Compose([
#     transforms.Resize((224, 224), antialias=True),
#     transforms.ToTensor(),
#     ...
# ])

# ——以下是你原来的代码——

def evaluate_video_folder(
    img_folder,
    anno_json_path,
    model_name, ckpt_path,
    proximity_thresh_px=20,
    out_csv="results.csv"
):
    # …（模型加载等）…

    # 加载标注
    with open(anno_json_path, "r") as f:
        annotations = json.load(f)

    records = []
    # 2. 在这里包一层 tqdm，就能看到进度条了
    for frame_name, persons in tqdm(annotations.items(), desc="Processing frames"):
        img_path = os.path.join(img_folder, frame_name)
        if not os.path.isfile(img_path):
            continue
        img_np = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)

        for p in persons:
            # …（你原有的 per-person 逻辑）…
            # 计算并保存 auc, avg_l2, min_l2
            pass

if __name__ == "__main__":
    evaluate_video_folder(
        img_folder         = r"C:\Users\qians\Desktop\gazelle-main\002",
        anno_json_path     = r"C:\Users\qians\Desktop\gazelle-main\002_GT_VideoFormat.json",
        model_name         = "gazelle_dinov2_vitl14",
        ckpt_path          = r"C:\Users\qians\Desktop\gazelle-main\gazelle_dinov2_vitl14.pt",
        proximity_thresh_px=20,
        out_csv            = "./gaze_metrics.csv"
    )