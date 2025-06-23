import os
import urllib
import json
def download_and_load_file(file_path, url):
    if not os.path.exists(file_path):
        with urllib.request.urlopen(url) as response:
            text_data = response.read().decode('utf-8')
        with open(file_path, 'w', encoding='utf-8') as file:
            file.write(text_data)
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

if __name__ == "__main__":
    file_path = 'instruction-data.json'
    url = 'https://github.com/jenhy/LLMs/tree/master/ch07/instruction-data.json'
    data = download_and_load_file(file_path, url)
    print(data[:10])
    print("Number of entries:", len(data))