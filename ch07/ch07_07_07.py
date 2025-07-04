import os,sys,json

import numpy as np
import tensorflow as tf
from tqdm import tqdm

import torch
from torch.utils.data import Dataset
from functools import partial
from torch.utils.data import DataLoader
import tiktoken
import time
import re

# Python 的​​进度条工具,用于在循环或长时间运行的任务中显示进度条，让用户直观地了解程序运行进度、剩余时间和速度等信息。
# 名字来源于阿拉伯语 ​​"taqaddum" ,意思是“进步”或“进展”（英文：progress）
from tqdm import tqdm   


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
from ch05.ch05_02_01 import plot_losses
from ch07.ch07_07_08 import query_model

def load_gpt2_params_from_tf_ckpt(ckpt_path, settings):
    # Initialize parameters dictionary with empty blocks for each layer
    params = {"blocks": [{} for _ in range(settings["n_layer"])]}

    # Iterate over each variable in the checkpoint
    for name, _ in tf.train.list_variables(ckpt_path):
        # Load the variable and remove singleton dimensions
        variable_array = np.squeeze(tf.train.load_variable(ckpt_path, name))

        # Process the variable name to extract relevant parts
        variable_name_parts = name.split("/")[1:]  # Skip the 'model/' prefix

        # Identify the target dictionary for the variable
        target_dict = params
        if variable_name_parts[0].startswith("h"):
            layer_number = int(variable_name_parts[0][1:])
            target_dict = params["blocks"][layer_number]

        # Recursively access or create nested dictionaries
        for key in variable_name_parts[1:-1]:
            target_dict = target_dict.setdefault(key, {})

        # Assign the variable array to the last key
        last_key = variable_name_parts[-1]
        target_dict[last_key] = variable_array

    return params
def download_and_load_gpt2(model_size, models_dir):
    # 支持的模型大小
    allowed_sizes = ("124M", "355M", "774M", "1558M")
    if model_size not in allowed_sizes:
        raise ValueError(f"Model size must be one of {allowed_sizes}")

    # 尝试在 Kaggle Dataset 中查找模型路径
    kaggle_dataset_path = f"/kaggle/input/gpt2-{model_size.lower()}-model/{models_dir}/{model_size}"
    if os.path.exists(kaggle_dataset_path):
        print(f"使用 Kaggle Dataset 中的模型：{kaggle_dataset_path}")
        model_dir = kaggle_dataset_path
    else:
        raise FileNotFoundError(
            f"未在 Kaggle Dataset 中找到模型目录：{kaggle_dataset_path}\n"
            "请确保已将 gpt2-xxx-model 数据集添加到 Notebook。"
        )

    # 读取 hparams 设置
    hparams_path = os.path.join(model_dir, "hparams.json")
    settings = json.load(open(hparams_path, "r", encoding="utf-8"))

    # 读取 TensorFlow checkpoint 参数
    tf_ckpt_path = tf.train.latest_checkpoint(model_dir)
    if tf_ckpt_path is None:
        raise FileNotFoundError("未能找到 TensorFlow checkpoint 文件（model.ckpt.*）")

    params = load_gpt2_params_from_tf_ckpt(tf_ckpt_path, settings)

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
    load_weights_into_gpt(model, params)
    model.to(device)
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

    # model_input = format_input(data[50])
    # desired_response = f"\n\n### Response:\n{data[50]['output']}"
    # print(model_input + desired_response)

    train_data, test_data, val_data = split_dataset(data, 0.85, 0.1)

    train_dataset = InstructionDataset(train_data, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=customized_collate_fn, shuffle=True, drop_last=True, num_workers=num_workers)

    test_dataset = InstructionDataset(test_data, tokenizer)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=customized_collate_fn, shuffle=False, drop_last=False, num_workers=num_workers)

    val_dataset = InstructionDataset(val_data, tokenizer)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=customized_collate_fn, shuffle=False, drop_last=False, num_workers=num_workers)

    # print("Train loader:")
    # for inputs, targets in train_loader:
    #     # 输入批次的形状为(batch_size, seq_len)，目标批次的形状为(batch_size, seq_len)。每一批可能有不同的长度。
    #     # 如：
    #     # torch.Size([8, 61]) torch.Size([8, 61])
    #     # torch.Size([8, 76]) torch.Size([8, 76])
    #     print(inputs.shape, targets.shape)

    # with torch.no_grad():
    #     train_loss = calc_loss_loader(train_loader, model, device)
    #     val_loss = calc_loss_loader(val_loader, model, device)
    # print("Training loss:", train_loss)
    # print("Validation loss:", val_loss)

    start_time = time.time()
    torch.manual_seed(123)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.00005, weight_decay=0.01)
    num_epochs = 2
    train_losses, val_losses, track_tokens_seen = train_model_simple(model, train_loader, val_loader, optimizer, device, num_epochs=num_epochs, eval_freq=5, eval_iter=5, start_context=format_input(val_data[0]), tokenizer=tokenizer)
    end_time = time.time()
    execution_time_minutes = (end_time - start_time) / 60
    print(f"Execution time: {execution_time_minutes:.2f} minutes.")

    epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
    plot_losses(epochs_tensor, track_tokens_seen, train_losses, val_losses)


    for entry in test_data[:3]:
        input_text = format_input(entry)
        token_ids = generate(model=model, idx=text_to_token_ids(input_text, tokenizer).to(device), max_new_tokens=256, context_size=BASE_CONFIG['context_length'], eos_id=50256)
        generated_text = token_ids_to_text(token_ids, tokenizer)

        response_text = generated_text[len(input_text):].replace("### Response:", "").strip()
        print(input_text)
        print(f"\nCorrect response:\n>> {entry['output']}")
        print(f"\nModel response:\n>> {response_text.strip()}\n")
        print("---------------------------")

    for i, entry in tqdm(enumerate(test_data), total=len(test_data)):
        input_text = format_input(entry)

        token_ids = generate(model=model, idx=text_to_token_ids(input_text, tokenizer).to(device), max_new_tokens=256, context_size=BASE_CONFIG['context_length'], eos_id=50256)
        generated_text = token_ids_to_text(token_ids, tokenizer)
        response_text = generated_text[len(input_text):].replace("### Response:", "").strip()
        test_data[i]["model_response"] = response_text

    with open("instruction-data-with-response.json", "w") as file:
        json.dump(test_data, file, indent=4)

    print(test_data[0])

    file_name = f"{re.sub(r'[ ()]', '', CHOOSE_MODEL)}-sft.pth"
    torch.save(model.state_dict(), file_name)
    print(f"Model saved as {file_name}")

    for entry in test_data[:3]:
        prompt = (
            f"Given the input '{format_input(entry)}' "
            f"and conrrect output '{entry['output']}', "
            f"score the model response '{entry['model_response']}'"
            f" on a scale from 0 to 100, where 100 is the best score. "
        )
        print("\nDataset response:")
        print(">>", entry['output'])
        print("\nModel response:")
        print(">>", entry["model_response"])
        print("\nScore:")
        print(">>", query_model(prompt))
        print("\n---------------------------------------")