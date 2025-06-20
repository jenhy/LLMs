import urllib.request
import re
from ch02_03 import SimpleTokenizerV1
from ch02_04 import SimpleTokenizerV2

# 定义代理地址
# proxy_url = "http://127.0.0.1:50383"
# # 创建代理处理器
# proxy_handler = urllib.request.ProxyHandler({
#     'http': proxy_url,
#     'https': proxy_url
# })

# # 创建一个 OpenerDirector 对象
# opener = urllib.request.build_opener(proxy_handler)

# # 安装 opener 为全局的 opener
# urllib.request.install_opener(opener)

# # 定义要下载的文件的 URL
# # url = "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch02/01_main-chapter-code/the-verdict.txt"
# url = ("https://raw.githubusercontent.com/rasbt/"
#        "LLMs-from-scratch/main/ch02/01_main-chapter-code/"
#        "the-verdict.txt")
# 定义本地保存文件的路径
file_path = f"C:\\Users\\Jenhy\\OneDrive\\doc\\学习\\AI\\LLMs\\ch02\\the-verdict.txt"

# try:
#     # 使用 urlretrieve 下载文件
#     urllib.request.urlretrieve(url, file_path)
#     print(f"文件下载成功，保存路径为: {file_path}")
# except urllib.error.URLError as e:
#     print(f"下载文件时发生网络错误: {e}")
# except Exception as e:
#     print(f"发生未知错误: {e}")
    
with open(file_path, 'r', encoding='utf-8') as f:
    raw_text = f.read()

print("Total number of characters: ",len(raw_text))
print(raw_text[:99])

preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
preprocessed = [item.strip() for item in preprocessed if item.strip()]
print(len(preprocessed))

# print(preprocessed[:30])

all_words = sorted(list(set(preprocessed)))

# 加入两个特殊的词元：<unk> 和 <|endoftext|>
all_words.extend(["<|endoftext|>","<|unk|>"])

vocab_size = len(all_words)
print(vocab_size)

vocab = {token: integer for integer, token in enumerate(all_words)}

print(len(vocab.items()))
for i, item in enumerate(vocab.items()):
    print(i, item)
    if i > 5000:
        break
    
tokenizer = SimpleTokenizerV1(vocab)

text = """"It's the last he painted, you know,"
Mrs. Gisburn said with pardonable pride.
"""
ids = tokenizer.encode(text)
print(ids)

print(tokenizer.decode(ids))

# 训练新样本
text = "Hello, do you like tea?"
# print(tokenizer.encode(text))

for i, item in enumerate(list(vocab.items())[-5:]):
    print(i, item)


text1 = "Hello, do you like tea?"
text2 = "In the sunlit terraces of the palace."

# 将两个句子连接起来,用|endoftext|作为分隔符
# 其中join方法的基本语法是：separator.join(iterable),separator：用于连接可迭代对象中元素的字符串。
# iterable：可迭代对象，比如列表、元组、字符串等等。此处(text1, text2) 是一个元组，属于可迭代对象。
text = " <|endoftext|> ".join((text1, text2))   
print(text)

tokenizer = SimpleTokenizerV2(vocab)
print(tokenizer.encode(text))

print(tokenizer.decode(tokenizer.encode(text)))

enc_text = tokenizer.encode(raw_text)
print(len(enc_text))