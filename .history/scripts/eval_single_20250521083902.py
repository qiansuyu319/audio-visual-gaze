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
        img_folder         = "./frames",
        anno_json_path     = "./video_annotations.json",
        model_name         = "gazelle_dinov2_vitl14",
        ckpt_path          = r"C:\Users\qians\Desktop\gazelle-main\gazelle_dinov2_vitl14.pt",
        proximity_thresh_px=20,
        out_csv            = "./gaze_metrics.csv"
    )
改动要点

eval_one_sample 中，不再对 avg_l2、min_l2 调用 .item()；而是直接将它们与 auc 一起 float(...)，确保返回的都是 Python float。

在主循环里对这三个值进行 round(..., 4) 并存入结果列表。

这样就能避免 “'float' object has no attribute 'item'” 的报错了。







您说：
andomResizedCrop(), etc.) will change from None to True in v0.17, in order to be consistent across the PIL and Tensor backends. To suppress this warning, directly pass antialias=True (recommended, future default), antialias=None (current default, which means False for Tensors and True for PIL), or antialias=False (only works on Tensors - PIL will still use antialiasing). This also applies if you are using the inference transforms from the models weights: update the call to weights.transforms(antialias=True).
  warnings.warn(
C:\ProgramData\anaconda3\Lib\site-packages\torchvision\transforms\functional.py:1603: UserWarning: The default value of the antialias parameter of all the resizing transforms (Resize(), RandomResizedCrop(), etc.) will change from None to True in v0.17, in order to be consistent across the PIL and Tensor backends. To suppress this warning, directly pass antialias=True (recommended, future default), antialias=None (current default, which means False for Tensors and True for PIL), or antialias=False (only works on Tensors - PIL will still use antialiasing). This also applies if you are using the inference transforms from the models weights: update the call to weights.transforms(antialias=True).
  warnings.warn( 一直出现这个 能不能给我一个进度条


ChatGPT 说：

已思考 14 秒
你可以通过两步来解决这个问题：

Suppress 警告
将 antialias 参数显式地传给你的 resize/crop transform，或者在脚本开头直接屏蔽这条 warning。

加上进度条
用 tqdm 给帧迭代加个进度条。

下面给出一个示例，在你的 evaluate_video_folder 前面做修改——只要把这些代码插到脚本最上面，然后在循环里用 tqdm 包裹就行了：

python
复制
编辑
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

    # 最后写 CSV
    # …

if __name__ == "__main__":
    evaluate_video_folder(
        img_folder         = r"C:\Users\qians\Desktop\gazelle-main\002",
        anno_json_path     = r"C:\Users\qians\Desktop\gazelle-main\002_GT_VideoFormat.json",
        model_name         = "gazelle_dinov2_vitl14",
        ckpt_path          = r"C:\Users\qians\Desktop\gazelle-main\gazelle_dinov2_vitl14.pt",
        proximity_thresh_px=20,
        out_csv            = "./gaze_metrics.csv"
    )
