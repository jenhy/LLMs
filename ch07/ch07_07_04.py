import torch
from torch.utils.data import Dataset
from functools import partial
from torch.utils.data import DataLoader
import tiktoken

import os, sys
# 添加上级路径以便导入自定义模块
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)  
sys.path.append(parent_dir)

from ch07.ch07_07_02 import download_and_load_file, format_input, split_dataset
from ch07.ch07_07_03 import custom_collate_fn
from ch07.ch07_07_03 import InstructionDataset

import os
# 设置代理
os.environ["HTTPS_PROXY"] = "http://127.0.0.1:49845"


if __name__ == "__main__":
    num_workers = 0
    batch_size = 8

    torch.manual_seed(123)

    tokenizer = tiktoken.get_encoding("gpt2")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # 函数包装器
    customized_collate_fn = partial(custom_collate_fn, device=device, allowed_max_length=1024)

    file_path = 'instruction-data.json'

    # https://github.com/jenhy/LLMs/tree/master/ch07/instruction-data.json不是一个原始的JSON文件链接，返回的是HTML页面，下载原始JSON文件内容改成raw链接形式
    url = 'https://raw.githubusercontent.com/jenhy/LLMs/master/ch07/instruction-data.json'
    data = download_and_load_file(file_path, url)
    # print("Example entry:\n", data[:2])
    # print("Another example entry:\n", data[99])
    # print("Number of entries:", len(data))

    model_input = format_input(data[50])
    desired_response = f"\n\n### Response:\n{data[50]['output']}"
    # print(model_input + desired_response)

    train_data, test_data, val_data = split_dataset(data, 0.85, 0.1)

    train_dataset = InstructionDataset(train_data, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=custom_collate_fn, shuffle=True, drop_last=True, num_workers=num_workers)

    test_dataset = InstructionDataset(test_data, tokenizer)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=custom_collate_fn, shuffle=False, drop_last=False, num_workers=num_workers)

    val_dataset = InstructionDataset(val_data, tokenizer)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=custom_collate_fn, shuffle=False, drop_last=False, num_workers=num_workers)

    print("Train loader:")
    for inputs, targets in train_loader:
        # 输入批次的形状为(batch_size, seq_len)，目标批次的形状为(batch_size, seq_len)。每一批可能有不同的长度。
        # 如：
        # torch.Size([8, 61]) torch.Size([8, 61])
        # torch.Size([8, 76]) torch.Size([8, 76])
        print(inputs.shape, targets.shape)