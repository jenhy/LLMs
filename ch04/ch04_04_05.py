import torch
import torch.nn as nn

import os, sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)  
sys.path.append(parent_dir)

from ch03.ch03_06_02 import MultiHeadAttention
from ch04.ch04_04_03 import FeedForward
from ch04.ch04_04_02 import LayerNorm

class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(d_in=cfg["emb_dim"], d_out=cfg["emb_dim"], context_length=cfg["context_length"], num_heads=cfg["n_heads"], dropout=cfg["drop_rate"], qkv_bias=cfg["qkv_bias"])
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])


    def forward(self, x):
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        return x

# 定义字典，用于配置GPT-2模型参数
GPT_CONFIG_124M = {
    "vocab_size": 50257,    # 词汇表大小，被BPE分词器使用的由50257个单词组成的词汇表
    "context_length": 1024, # 上下文长度，能够处理的最大输入词元数量
    "emb_dim": 768,         # 嵌入维度，每个词元转化为768维向量
    "n_heads": 12,          # 多头注意力机制中注意力头的数量
    "n_layers": 12,         # Transformer块的数量
    "drop_rate": 0.1,       # 丢弃率10%，在训练过程中，随机丢弃一些神经元，以减少过拟合
    "qkv_bias": False       # 是否使用偏置项。在多头注意力机制的线性层中添加一个偏置项，用于查询、键和值的计算
}

if __name__ == "__main__":
    torch.manual_seed(123)
    x = torch.rand(2, 4, 768)
    block = TransformerBlock(GPT_CONFIG_124M)
    output = block(x)

    # Input shape: torch.Size([2, 4, 768])
    # Output shape: torch.Size([2, 4, 768])
    print("Input shape:", x.shape)
    print("Output shape:", output.shape)