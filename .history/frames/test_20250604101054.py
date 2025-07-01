import os
import re

def compress_filenames(folder_path, extension=".jpg"):
    pattern = re.compile(r"^(\d{3})(\d{5})" + re.escape(extension) + r"$")  # 匹配 04900001.jpg

    for filename in os.listdir(folder_path):
        match = pattern.match(filename)
        if match:
            prefix, number = match.groups()
            number_trimmed = str(int(number))  # 去除前导0，例如 00001 → 1
            new_name = f"{prefix}{number_trimmed.zfill(3)}{extension}"  # 补足后缀位数为至少3位
            src = os.path.join(folder_path, filename)
            dst = os.path.join(folder_path, new_name)
            os.rename(src, dst)
            print(f"✅ {filename} → {new_name}")
        else:
            print(f"⏭️ Skipped: {filename}")

    print("🎉 All filenames compressed to 6-digit format.")

if __name__ == "__main__":
    compress_filenames(
        folder_path=r"C:\Users\qians\Desktop\gazelle-main\frames",  # 修改为你的路径
        extension=".jpg"
    )
