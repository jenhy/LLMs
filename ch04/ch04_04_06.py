import torch
import torch.nn as nn
import tiktoken
import os, sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)  
sys.path.append(parent_dir)

from ch04.ch04_04_05 import TransformerBlock
from ch04.ch04_04_02 import LayerNorm

class GPTModel(nn.Module):
    def __init__(self, cfg):
        #  初始化构造函数，传入cfg（配置字典），调用父类的初始化。
        #  调用了父类 nn.Module 的构造器，这一步是 必须的，否则你的模型无法成为一个 PyTorch 模块。
        #  主要负责做两件事：注册模型中的子模块（子层）；初始化模块的状态字典、参数、缓冲区等。
        super().__init__()

        #  词元嵌入
        #  词元（Token）嵌入层：把每个词元的整数 ID 转换为一个 emb_dim 维的向量。
        #  输入为一批词元id，输出为 [vocab_size, emb_dim]。如：torch.Size([50257, 768])
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        # print(self.tok_emb.weight.shape)

        #  位置嵌入
        #  位置嵌入层：把每个位置（最多 context_length 个）转换为一个 emb_dim 维的向量。
        #  输入为 [seq_len]，输出为 [context_length, emb_dim]。如：torch.Size([1024, 768])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        # print(self.pos_emb.weight.shape)

        #  dropout隐藏单元被丢弃率
        #  Dropout 层：防止过拟合。每次训练随机丢弃部分节点（神经元）。
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        self.trf_blocks = nn.Sequential(*[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])

        #  最终层归一化
        self.final_norm = LayerNorm(cfg["emb_dim"])

        #  线性输出层，将嵌入词元向量映射到词汇空间
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx):
        device = self.tok_emb.weight.device
        in_idx = in_idx.to(device)
        print(f"in_idx.device:{in_idx.device}")
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)

        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))

        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)   # 这里我误写成了final_norm(x)，导致输出的结果为768维向量，而不是vocab_size维向量

        return logits
    

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

## 练习题4.2
# 定义字典，用于配置GPT-2模型参数
# GPT_CONFIG_124M = {
#     "vocab_size": 50257,    # 词汇表大小，被BPE分词器使用的由50257个单词组成的词汇表
#     "context_length": 1024, # 上下文长度，能够处理的最大输入词元数量
#     "emb_dim": 1024,         # 嵌入维度，每个词元转化为1024维向量
#     "n_heads": 16,          # 多头注意力机制中注意力头的数量
#     "n_layers": 24,         # Transformer块的数量
#     "drop_rate": 0.1,       # 丢弃率10%，在训练过程中，随机丢弃一些神经元，以减少过拟合
#     "qkv_bias": False       # 是否使用偏置项。在多头注意力机制的线性层中添加一个偏置项，用于查询、键和值的计算
# }


if __name__ == "__main__":
    torch.manual_seed(123)
    model = GPTModel(GPT_CONFIG_124M)

    """
    下载GPT-2分词器资源文件时需要翻墙访问，所以需要先设置代理，端口查看Clash软件
    $env:HTTPS_PROXY="http://127.0.0.1:50071"
    $env:HTTP_PROXY="http://127.0.0.1:50071"
    """
    tokenizer = tiktoken.get_encoding("gpt2")
    batch = []
    txt1 = "Every effort moves you"
    txt2 = "Every day holds a"
    batch.append(torch.tensor(tokenizer.encode(txt1), dtype=torch.long))
    batch.append(torch.tensor(tokenizer.encode(txt2), dtype=torch.long))

    """torch.stack 这个函数名中的 "stack" 是英文单词，意思是 “堆叠”、“叠放”，它的功能也正如其名：将多个张量沿着一个新维度“堆叠”在一起，生成一个新的更高维的张量。
    注意：这里我犯了一个错误，没有将stack的结果保存到batch中，导致后面的forward操作出错了
    batch_size, seq_len = in_idx.shape
    AttributeError: 'list' object has no attribute 'shape'
    """
    batch = torch.stack(batch, dim=0)

    out = model(batch)
    print("Input shape:\n", batch)  # Output shape: torch.Size([2, 4, 50257])
    """tensor([[[ 0.1381,  0.0077, -0.1963,  ..., -0.0222, -0.1060,  0.1717],
         [ 0.3865, -0.8408, -0.6564,  ..., -0.5163,  0.2369, -0.3357],
         [ 0.6989, -0.1829, -0.1631,  ...,  0.1472, -0.6504, -0.0056],
         [-0.4290,  0.1669, -0.1258,  ...,  1.1579,  0.5303, -0.5549]],

        [[ 0.1094, -0.2894, -0.1467,  ..., -0.0557,  0.2911, -0.2824],
         [ 0.0882, -0.3552, -0.3527,  ...,  1.2930,  0.0053,  0.1898],
         [ 0.6091,  0.4702, -0.4094,  ...,  0.7688,  0.3787, -0.1974],
         [-0.0612, -0.0737,  0.4751,  ...,  1.2463, -0.3834,  0.0609]]],
       grad_fn=<UnsafeViewBackward0>)"""
    print("\nOutput shape:", out.shape)
    print(out)


    # 统计模型参数张量的总参数量
    # 通过model.parameters()方法直接访问模型的所有可训练参数
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")    # Total parameters: 163,009,536

    print("Token embedding layer shape:", model.tok_emb.weight.shape)
    print("Output layer shape:", model.out_head.weight.shape)

    # 权重共享。词元嵌入层和输出层的权重使用同一份参数。
    # 从总的GPT-2模型参数计数中减去输出层参数的数量，得到GPT-2模型参数的数量1.24亿。
    total_params_gpt2 = total_params - sum(p.numel() for p in model.out_head.parameters())
    print(f"Number of trainable parameters "
          f"considering weight tring:{total_params_gpt2:,}")    # Number of trainable parameters considering weight tring:124,412,160
    

    ## 练习题4.1
    # 计算前馈模块所包含的参数数量
    total_params_ff = sum(p.numel() for p in model.trf_blocks[0].ff.parameters())
    # Total parameters in the first Transformer block's feed-forward module: 4,722,432
    print(f"Total parameters in the first Transformer block's feed-forward module: {total_params_ff:,}")

    # 计算注意力模块所包含的参数数量
    total_params_att = sum(p.numel() for p in model.trf_blocks[0].att.parameters())
    # Total parameters in the first Transformer block's attention module: 2,360,064
    print(f"Total parameters in the first Transformer block's attention module: {total_params_att:,}")

    # 计算GPTModel中1.63亿个参数的内存需求
    total_size_bytes = total_params * 4 # 假设每个参数占用4字节内存，32位浮点数
    total_size_mb = total_size_bytes / (1024 ** 2)
    print(f"Memory requirement for 1.63 billion parameters: {total_size_mb:.2f} MB")    # Memory requirement for 1.63 billion parameters: 621.83 MB

    