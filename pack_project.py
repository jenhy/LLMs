import os
import zipfile

def zip_dir(dir_path, zip_path, exclude_dirs=None):
    exclude_dirs = exclude_dirs or []

    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(dir_path):
            # 跳过被排除的目录
            if any(excluded in root for excluded in exclude_dirs):
                continue

            for file in files:
                filepath = os.path.join(root, file)
                arcname = os.path.relpath(filepath, start=dir_path)
                zipf.write(filepath, arcname)

if __name__ == "__main__":
    project_dir = r"G:\我的云端硬盘\LLMs"
    zip_path = r"G:\我的云端硬盘\LLMs.zip"
    
    # 设置排除路径，可以是完整路径的一部分
    exclude = ['gpt2', '.git', '__pycache__']

    zip_dir(project_dir, zip_path, exclude_dirs=exclude)
    print("已打包（排除指定目录）:", zip_path)
