import urllib.request
import psutil
import json
import urllib
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

if __name__ == "__main__":
    ollama_running = check_if_running("ollama")

    if not ollama_running:
        raise RuntimeError("Ollama is not running. Launch ollama before proceeding.")

    print("Ollama is running:", check_if_running("ollama"))

    model = "tinyllama:latest"
    result = query_model("What do Llamas eat?", model)
    print(result)

    