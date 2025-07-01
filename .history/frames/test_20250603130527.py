import os

# 设置路径和前缀
folder = r"C:\Users\qians\Desktop\gazelle-main\frames"  # 修改为你的帧文件夹路径
video_id = "066"  # 你希望保留的前缀编号
digits = 3        # 帧号保留的位数，如 000、001、...

# 获取并排序所有 jpg 文件
files = sorted([f for f in os.listdir(folder) if f.endswith(".jpg")])

for idx, filename in enumerate(files):
    new_name = f"{video_id}{str(idx).zfill(digits)}.jpg"
    old_path = os.path.join(folder, filename)
    new_path = os.path.join(folder, new_name)
    os.rename(old_path, new_path)
    print(f"✅ Renamed: {filename} → {new_name}")
