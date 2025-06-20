import urllib.request

# 定义代理地址
proxy_url = "http://127.0.0.1:50383"
# 创建代理处理器
proxy_handler = urllib.request.ProxyHandler({
    'http': proxy_url,
    'https': proxy_url
})

# 创建一个 OpenerDirector 对象
opener = urllib.request.build_opener(proxy_handler)

# 安装 opener 为全局的 opener
urllib.request.install_opener(opener)

# 定义要下载的文件的 URL
# url = "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch02/01_main-chapter-code/the-verdict.txt"
url = ("https://raw.githubusercontent.com/rasbt/"
       "LLMs-from-scratch/main/ch02/01_main-chapter-code/"
       "the-verdict.txt")
# 定义本地保存文件的路径
file_path = "the-verdict.txt"

try:
    # 使用 urlretrieve 下载文件
    urllib.request.urlretrieve(url, file_path)
    print(f"文件下载成功，保存路径为: {file_path}")
except urllib.error.URLError as e:
    print(f"下载文件时发生网络错误: {e}")
except Exception as e:
    print(f"发生未知错误: {e}")
    
with open(file_path, 'r', encoding='utf-8') as f:
    raw_text = f.read()
print("Total number of characters: ",len(raw_text))
print(raw_text[:99])