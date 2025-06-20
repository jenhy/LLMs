import torch
import tiktoken

tokenizer = tiktoken.get_encoding("gpt2")
inputs = tokenizer.encode("Do you have time")

# torch.Size([4])   单条文本的 token 序列（向量，也就是一维张量）
# torch.Size([1, 4])    模拟 batch 输入，batch size = 1(二维张量)
print(torch.tensor(inputs).shape)
print(torch.tensor(inputs).unsqueeze(0).shape)

