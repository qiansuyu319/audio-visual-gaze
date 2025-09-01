import os
import json
import pandas as pd
from PIL import Image
import argparse

def detect_offset(df, img_root, video_id):
    """检测 GT 与实际帧文件的起始差值"""
    # GT 最小帧号
    gt_min = min([int(str(name).replace(".jpg", "").replace(".jpg", "")[-6:]) for name in df["frame_name"]])

    # 实际帧目录的最小帧号
    img_dir = os.path.join(img_root, video_id)
    img_files = sorted([f for f in os.listdir(img_dir) if f.endswith(".jpg")])
    if not img_files:
        return 0
    img_min = int(img_files[0].replace(".jpg", "")[-6:])

    offset = img_min - gt_min
    if offset != 0:
        print(f"[INFO] Detected frame offset for video {video_id}: {offset}")
    return offset


def process_single_csv(csv_file, img_root):
    video_id = os.path.basename(csv_file).split("_")[0]  # e.g. "002"
    frames = []

    df = pd.read_csv(csv_file)
    # 一次性检测 offset
    offset = detect_offset(df, img_root, video_id)

    for _, row in df.iterrows():
        # 清理 GT 中的名字
        img_name = str(row["frame_name"]).replace(".jpg.jpg", ".jpg")
        frame_id = int(img_name.replace(".jpg", "")[-6:])
        frame_id = frame_id + offset   # 统一应用 offset
        img_name = f"{frame_id:06d}.jpg"
        img_path = os.path.join(img_root, video_id, img_name)

        if not os.path.exists(img_path):
            print(f"[WARN] missing image {img_path}")
            continue

        try:
            with Image.open(img_path) as im:
                width, height = im.size
        except Exception as e:
            print(f"[WARN] cannot open {img_path}: {e}")
            continue

        head = {
            "bbox": [int(row["head_xmin"]), int(row["head_ymin"]), int(row["head_xmax"]), int(row["head_ymax"])],
            "bbox_norm": [int(row["head_xmin"]) / width, int(row["head_ymin"]) / height,
                          int(row["head_xmax"]) / width, int(row["head_ymax"]) / height],
            "gazex": [int(row["gaze_x"])],
            "gazey": [int(row["gaze_y"])],
            "gazex_norm": [int(row["gaze_x"]) / width],
            "gazey_norm": [int(row["gaze_y"]) / height],
            "inout": 1,
            "head_id": int(row["person_id"])
        }

        frame = {
            "path": f"{video_id}/{img_name}",
            "heads": [head],
            "num_heads": 1,
            "width": width,
            "height": height
        }
        frames.append(frame)

    print(f"[OK] {os.path.basename(csv_file)} -> {len(frames)} frames (video {video_id})")
    return frames


def batch_convert(csv_root, img_root, out_json):
    all_data = []
    for file in sorted(os.listdir(csv_root)):
        if not file.endswith(".csv"):
            continue
        if "all" in file.lower():  # 跳过汇总文件
            print(f"[SKIP] {file}")
            continue

        csv_path = os.path.join(csv_root, file)
        frames = process_single_csv(csv_path, img_root)
        all_data.extend(frames)

    with open(out_json, "w") as f:
        json.dump(all_data, f)

    print(f"[DONE] saved {out_json}, total {len(all_data)} frames")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_root", type=str, required=True, help="CSV 文件目录")
    parser.add_argument("--img_root", type=str, required=True, help="frames 根目录")
    parser.add_argument("--out_json", type=str, default="train_preprocessed.json", help="输出 JSON")
    args = parser.parse_args()

    batch_convert(args.csv_root, args.img_root, args.out_json)
