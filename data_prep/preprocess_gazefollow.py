import os
import json
import pandas as pd
from PIL import Image
import argparse
from collections import defaultdict

parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, default="./data/vgs")  # 存放 CSV 和 frames 的根目录
parser.add_argument("--csv_train", type=str, default="GT_CSV")
parser.add_argument("--csv_test", type=str, default="test_GT.csv")
parser.add_argument("--frame_root", type=str, default="frames")
args = parser.parse_args()


def process_csv(csv_path, img_root, is_train=True):
    df = pd.read_csv(csv_path)

    FRAME_DICT = defaultdict(list)
    for _, row in df.iterrows():
        img_name = str(row["frame_name"]).replace(".jpg.jpg", ".jpg")
        FRAME_DICT[img_name].append(row)

    multiperson_ex = 0
    FRAMES = []

    for img_name, rows in FRAME_DICT.items():
        video_id = img_name[:3]  # 前三位是 video_id
        img_path = os.path.join(img_root, video_id, img_name)
        if not os.path.exists(img_path):
            print(f"[WARN] missing image {img_path}")
            continue

        with Image.open(img_path) as im:
            width, height = im.size

        num_people = len(rows)
        if num_people > 1:
            multiperson_ex += 1

        heads = []
        for i, row in enumerate(rows):
            xmin, ymin, xmax, ymax = int(row["head_xmin"]), int(row["head_ymin"]), int(row["head_xmax"]), int(row["head_ymax"])
            gazex, gazey = int(row["gaze_x"]), int(row["gaze_y"])
            gazex_norm, gazey_norm = gazex / width, gazey / height

            if xmin > xmax:
                xmin, xmax = xmax, xmin
            if ymin > ymax:
                ymin, ymax = ymax, ymin

            xmin = max(xmin, 0)
            ymin = max(ymin, 0)
            xmax = min(xmax, width)
            ymax = min(ymax, height)

            head = {
                "bbox": [xmin, ymin, xmax, ymax],
                "bbox_norm": [xmin/width, ymin/height, xmax/width, ymax/height],
                "gazex": [gazex],
                "gazey": [gazey],
                "gazex_norm": [gazex_norm],
                "gazey_norm": [gazey_norm],
                "inout": 1 if is_train else 1,  # 测试集默认 in-frame
                "head_id": int(row["person_id"]) if "person_id" in row else i
            }
            if not is_train:
                head["num_annot"] = 1  # 测试集加 num_annot 字段
            heads.append(head)

        FRAMES.append({
            "path": os.path.join(video_id, img_name),
            "heads": heads,
            "num_heads": num_people,
            "width": width,
            "height": height
        })

    split = "Train" if is_train else "Test"
    print(f"{split} set: {len(FRAMES)} frames, {multiperson_ex} multi-person")
    return FRAMES


def main():
    train_csv_path = os.path.join(args.data_path, args.csv_train)
    test_csv_path = os.path.join(args.data_path, args.csv_test)
    frame_root = os.path.join(args.data_path, args.frame_root)

    TRAIN_FRAMES = process_csv(train_csv_path, frame_root, is_train=True)
    with open(os.path.join(args.data_path, "train_preprocessed.json"), "w") as f:
        json.dump(TRAIN_FRAMES, f)

    TEST_FRAMES = process_csv(test_csv_path, frame_root, is_train=False)
    with open(os.path.join(args.data_path, "test_preprocessed.json"), "w") as f:
        json.dump(TEST_FRAMES, f)


if __name__ == "__main__":
    main()
