import os,sys,json

import torch
from torch.utils.data import Dataset
from functools import partial
from torch.utils.data import DataLoader
import tiktoken

# 添加上级路径以便导入自定义模块
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)  
sys.path.append(parent_dir)

# from ch05.gpt_download import download_and_load_gpt2
from ch04.ch04_04_06 import GPTModel
from ch05.ch05_05_01 import model_configs, load_weights_into_gpt
from ch05.ch05_01_03 import calc_loss_loader
from ch05.ch05_02_01 import train_model_simple

from ch07.ch07_07_02 import download_and_load_file, format_input, split_dataset
from ch07.ch07_07_03 import custom_collate_fn
from ch07.ch07_07_03 import InstructionDataset
from ch05.ch05_03_03 import generate, text_to_token_ids, token_ids_to_text

def download_and_load_gpt2(model_size, models_dir):
    # 支持 Kaggle Dataset(改写版)
    allowed_sizes = ("124M", "355M", "774M", "1558M")
    if model_size not in allowed_sizes:
        raise ValueError(f"Model size must be in {allowed_sizes}")

    # Kaggle Dataset 路径
    kaggle_dataset_path = f"/kaggle/input/gpt2-355m-model/{models_dir}/{model_size}"
    if os.path.exists(kaggle_dataset_path):
        print(f"使用 Kaggle Dataset 中的模型：{kaggle_dataset_path}")
        model_dir = kaggle_dataset_path
    else:
        raise FileNotFoundError("未在 Kaggle Dataset 中找到模型目录。请确认数据集已添加。")

    # 你只需要 hparams.json 文件来读取结构参数
    hparams_path = os.path.join(model_dir, "hparams.json")
    settings = json.load(open(hparams_path, "r", encoding="utf-8"))

    # 返回设置，但不加载权重（因为你模型用的是 GPTModel，从头训练）
    params = None

    return settings, params

if __name__ == "__main__":
    BASE_CONFIG = {
        "vocab_size": 50257,
        "context_length": 1024,
        "drop_rate": 0.0,
        "qkv_bias": True
    }

    model_configs = {
        "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
        "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
        "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
        "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25}
    }

    CHOOSE_MODEL = "gpt2-medium (355M)"
    BASE_CONFIG.update(model_configs[CHOOSE_MODEL])
    model_size = CHOOSE_MODEL.split(" ")[-1].lstrip("(").rstrip(")")

    settings, params = download_and_load_gpt2(model_size=model_size, models_dir="gpt2")

    torch.manual_seed(123)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    model = GPTModel(BASE_CONFIG)
    model.to(device)

    load_weights_into_gpt(model, params)
    model.eval()

    num_workers = 0
    batch_size = 8


    tokenizer = tiktoken.get_encoding("gpt2")

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

    input_text = format_input(val_data[0])
    # print(input_text)

    token_ids = generate(model=model, idx=text_to_token_ids(input_text, tokenizer), max_new_tokens=35, context_size=BASE_CONFIG['context_length'], eos_id=50256)
    generated_text = token_ids_to_text(token_ids, tokenizer)

    response_text = generated_text[len(input_text):].strip()

    print(response_text)

    train_dataset = InstructionDataset(train_data, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=custom_collate_fn, shuffle=True, drop_last=True, num_workers=num_workers)

    test_dataset = InstructionDataset(test_data, tokenizer)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=custom_collate_fn, shuffle=False, drop_last=False, num_workers=num_workers)

    val_dataset = InstructionDataset(val_data, tokenizer)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=custom_collate_fn, shuffle=False, drop_last=False, num_workers=num_workers)

    # print("Train loader:")
    # for inputs, targets in train_loader:
    #     # 输入批次的形状为(batch_size, seq_len)，目标批次的形状为(batch_size, seq_len)。每一批可能有不同的长度。
    #     # 如：
    #     # torch.Size([8, 61]) torch.Size([8, 61])
    #     # torch.Size([8, 76]) torch.Size([8, 76])
    #     print(inputs.shape, targets.shape)

    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device)
        val_loss = calc_loss_loader(val_loader, model, device)
    print("Training loss:", train_loss)
    print("Validation loss:", val_loss)