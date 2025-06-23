import os
import json
import urllib.request
def download_and_load_file(file_path, url):
    """
    下载数据集
    """
    if not os.path.exists(file_path):
        with urllib.request.urlopen(url) as response:
            text_data = response.read().decode('utf-8')
        with open(file_path, 'w', encoding='utf-8') as file:
            file.write(text_data)
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

def format_input(entry):
    """
    实现提示词风格的输入(以Alpaca提示风格为例)
    """
    instruction_text = (
        f"Below is an instruction that describes a task. Write a response that appropriately completes the request."
        f"\n\n### Instruction:\n{entry['instruction']}"
    )

    input_text = (
        f"\n\n### Input:\n{entry['input']}" if entry["input"] else ""
    )

    return instruction_text + input_text

def random_split(df, train_frac, test_frac):
    """
    拆分数据集
    """

    #  计算拆分索引
    train_end = int(len(df) * train_frac)
    test_end = train_end + int(len(df) * test_frac)
    
    #  拆分数据集
    train_df = df[:train_end]
    test_df = df[train_end:test_end]
    val_df = df[test_end:]

    return train_df, test_df, val_df


if __name__ == "__main__":
    file_path = 'instruction-data.json'

    # https://github.com/jenhy/LLMs/tree/master/ch07/instruction-data.json不是一个原始的JSON文件链接，返回的是HTML页面，下载原始JSON文件内容改成raw链接形式
    url = 'https://raw.githubusercontent.com/jenhy/LLMs/master/ch07/instruction-data.json'
    data = download_and_load_file(file_path, url)
    print("Example entry:\n", data[:2])
    print("Another example entry:\n", data[99])
    print("Number of entries:", len(data))

    model_input = format_input(data[50])
    desired_response = f"\n\n### Response:\n{data[50]['output']}"
    print(model_input + desired_response)

    train_data, test_data, val_data = random_split(data, 0.85, 0.1)
    print("Number of training entries:", len(train_data))
    print("Number of validation entries:", len(val_data))
    print("Number of test entries:", len(test_data))