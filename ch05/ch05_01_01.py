import torch
import tiktoken
import os, sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)  
sys.path.append(parent_dir)

r"""current_dir: c:\Users\Jenhy\OneDrive\doc\学习\AI\LLMs\ch05, parent_dir: c:\Users\Jenhy\OneDrive\doc\学习\AI\LLMs, sys.path: ['c:\\Users\\Jenhy\\OneDrive\\doc\\学习\\AI\\LLMs\\ch05', 'D:\\Program Files\\Python\\Python38\\python38.zip', 'D:\\Program Files\\Python\\Python38\\DLLs', 'D:\\Program Files\\Python\\Python38\\lib', 'D:\\Program Files\\Python\\Python38', 'D:\\Program Files\\Python\\Python38\\lib\\site-packages', 'c:\\Users\\Jenhy\\OneDrive\\doc\\学习\\AI\\LLMs']"""
# print(f"current_dir: {current_dir}, parent_dir: {parent_dir}, sys.path: {sys.path}")

from ch04.ch04_04_06 import GPTModel
from ch04.ch04_04_07 import generate_text_simple

# 定义字典，用于配置GPT-2模型参数
GPT_CONFIG_124M = {
    "vocab_size": 50257,    # 词汇表大小，被BPE分词器使用的由50257个单词组成的词汇表
    "context_length": 256,  # 上下文长度，能够处理的最大输入词元数量，将1024个词元改成256个，减少训练模型的计算需求
    "emb_dim": 768,         # 嵌入维度，每个词元转化为768维向量
    "n_heads": 12,          # 多头注意力机制中注意力头的数量
    "n_layers": 12,         # Transformer块的数量
    "drop_rate": 0.1,       # 丢弃率10%，在训练过程中，随机丢弃一些神经元，以减少过拟合
    "qkv_bias": False       # 是否使用偏置项。在多头注意力机制的线性层中添加一个偏置项，用于查询、键和值的计算
}

torch.manual_seed(123)
model = GPTModel(GPT_CONFIG_124M)
model.eval()

def text_to_token_ids(text, tokenizer):
    r"""
    使用分词器将输入文本转换为词元ID。
    """
    encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)  # 使用unsqueeze()方法将输入张量扩展为二维张量，也就是添加batch维度
    return encoded_tensor
def token_ids_to_text(token_ids, tokenizer):
    r"""
    使用分词器将词元ID转换为文本。
    """
    flat = token_ids.squeeze(0)   # 使用squeeze()方法将输入张量压缩为1维张量，也就是删除batch维度
    return tokenizer.decode(flat.tolist())

if  __name__ == "__main__":
    start_context = "Every effort moves you"
    tokenizer = tiktoken.get_encoding("gpt2")

    token_ids = generate_text_simple(model=model, idx=text_to_token_ids(start_context, tokenizer), max_new_tokens=10, context_size=GPT_CONFIG_124M["context_length"])
    print("Output text:\n", token_ids_to_text(token_ids, tokenizer))

