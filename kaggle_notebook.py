!pip install gdown
import gdown
import zipfile
import sys

file_id = "1lILowh7vXZqNX8g39LuEYNpxfz1PO0hu"
output = "LLMs.zip"

gdown.download(f"https://drive.google.com/uc?id={file_id}", output, quiet=False)

with zipfile.ZipFile(output, "r") as zip_ref:
    zip_ref.extractall("project_code")

sys.path.append("project_code")


import zipfile

zip_path = "/kaggle/working/LLMs.zip"
extract_path = "/kaggle/working/LLMs"

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_path)


import os
os.chdir("/kaggle/working/LLMs")

# 单独开辟一个 Python 进程
!python ch06/ch06_06_07.py

# 脚本在当前 Python 进程中运行
%run ch06/ch06_06_07.py


"""https://drive.google.com/file/d/1lILowh7vXZqNX8g39LuEYNpxfz1PO0hu/view?usp=drive_link

import os

print("当前工作目录：", os.getcwd())
print("该目录下的文件：", os.listdir())
print("LLMs 目录下的文件：", os.listdir("/kaggle/working/LLMs"))


kaggle我的LLMs学习GPT2数据集：
www.kaggle.com/datasets/chenjianhai2025/gpt2-124m-model
"""

## 使用本地 VS Code + 远程 Kaggle开发环境
# 提交代码
git add .
git commit -m "fix: 更新模型结构/逻辑"
git push origin master


!git clone https://github.com/jenhy/LLMs.git

%cd LLMs

import os
os.chdir("/kaggle/working/LLMs")


# Auto clone or update repo
import os

REPO_NAME = "LLMs"
REPO_URL = "https://github.com/jenhy/LLMs.git"

if not os.path.exists(REPO_NAME):
    !git clone {REPO_URL}
else:
    %cd {REPO_NAME}
    !git pull origin master
    %cd ..
