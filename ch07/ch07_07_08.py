import urllib.request
import psutil
import json
import urllib
import os,sys
from tqdm import tqdm

# 添加上级路径以便导入自定义模块
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)  
sys.path.append(parent_dir)

from ch07.ch07_07_02 import download_and_load_file, format_input, split_dataset

def check_if_running(process_name):
    running = False
    for proc in psutil.process_iter(['name']):
        if process_name in proc.info['name']:  # Check if the process name contains the given string
            running = True
            break

    return running

def query_model(prompt, model="tinyllama", url="http://localhost:11434/api/generate"):
    """
    使用/api/generate
    """
    data = {
        "model": model,
        "prompt": prompt,
        "options": {
            "temperature": 0.0,
            "seed": 123,
            "num_ctx": 2048,
            "num_predict": 128  # 限制生成长度，防止卡死
        }
    }

    payload = json.dumps(data).encode("utf-8")
    request = urllib.request.Request(url, data=payload, method="POST")
    request.add_header("Content-Type", "application/json")

    response_data = ""
    with urllib.request.urlopen(request) as response:
        while True:
            line = response.readline()
            if not line:
                break
            line = line.decode("utf-8").strip()
            if not line:
                continue
            json_data = json.loads(line)
            response_data += json_data.get("response", "")

    return response_data

# def query_model(prompt, model="llama3", url="http://localhost:11434/api/chat"):
#     """
#     使用/api/chat
#     """
#     data = {
#         "model": model,
#         "messages":[
#             {"role": "user", "content": prompt}
#         ],
#         "options": {
#             "seed": 123,
#             "temperature": 0,
#             "num_ctx": 2048
#         }
#     }

#     payload = json.dumps(data).encode("utf-8")
#     request = urllib.request.Request(url, data=payload, method="POST")
#     request.add_header("Content-Type", "application/json")

#     with urllib.request.urlopen(request) as response:
#         while True:
#             line = response.readline().decode("utf-8")
#             if not line:
#                 break
#             response_json = json.load(line)
#             response_data += response_json["message"]["content"]
#     return response_data

def generate_model_scores(json_data, json_key, model="tinyllama"):
    scores = []
    for entry in tqdm(json_data, desc="Scoring entryies"):
        prompt = (
            f"Given the input '{format_input(entry)}' "
            f"and conrrect output '{entry['output']}', "
            f"score the model response '{entry[json_key]}'"
            f" on a scale from 0 to 100, where 100 is the best score. "
            f"Respond with the integer number only."
        )
        score = query_model(prompt, model)
        try:
            scores.append(int(score))
        except ValueError:
            print(f"Could not convert score: {score}")
            continue

    return scores

if __name__ == "__main__":
    ollama_running = check_if_running("ollama")

    if not ollama_running:
        raise RuntimeError("Ollama is not running. Launch ollama before proceeding.")

    print("Ollama is running:", check_if_running("ollama"))

    model = "tinyllama:latest"
    result = query_model("What do Llamas eat?", model)
    # print(result)

    file_path = 'instruction-data-with-response.json'

    # https://github.com/jenhy/LLMs/tree/master/ch07/instruction-data.json不是一个原始的JSON文件链接，返回的是HTML页面，下载原始JSON文件内容改成raw链接形式
    url = 'https://raw.githubusercontent.com/jenhy/LLMs/master/ch07/instruction-data-with-response.json'
    data = download_and_load_file(file_path, url)
    # print("Example entry:\n", data[:2])
    # print("Another example entry:\n", data[99])
    # print("Number of entries:", len(data))

    # model_input = format_input(data[50])
    # desired_response = f"\n\n### Response:\n{data[50]['output']}"
    # print(model_input + desired_response)

    train_data, test_data, val_data = split_dataset(data, 0.85, 0.1)

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

    scores = generate_model_scores(test_data, "model_response")
    print(f"Number of scores: {len(scores)} of {len(test_data)}")
    print(f"Average score: {sum(scores) / len(scores):.2f}")