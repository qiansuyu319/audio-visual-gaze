import os
import re

def rename_images_remove_underscore(folder_path, extension=".jpg"):
    pattern = re.compile(r"^(\d{3})_(\d+)" + re.escape(extension) + r"$")  # 匹配如 049_00594.jpg

    for filename in os.listdir(folder_path):
        match = pattern.match(filename)
        if match:
            prefix, number = match.groups()
            new_name = f"{prefix}{number}{extension}"  # 拼接成 049594.jpg
            src = os.path.join(folder_path, filename)
            dst = os.path.join(folder_path, new_name)
            os.rename(src, dst)
            print(f"✅ {filename} → {new_name}")
        else:
            print(f"⏭️ Skipped: {filename}")

    print("🎉 Underscore removal complete.")

if __name__ == "__main__":
    rename_images_remove_underscore(
        folder_path=r"C:\Users\qians\Desktop\gazelle-main\frames",  # 修改为你的实际路径
        extension=".jpg"
    )
