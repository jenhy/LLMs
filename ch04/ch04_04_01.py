import torch
import torch.nn as nn
import tiktoken

"""
定义一个简化版的GPT模型
继承自 PyTorch 的 nn.Module，表示是一个神经网络模块。
"""
class DummyGPTModel(nn.Module):
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

        #  定义多个Transformer块，n_layers表示有多个Transformer块
        """
        详细拆解代码：
        总体目的：构建一个由多个 DummyTransformerBlock 组成的序列模块，并将它包装成一个 nn.Sequential 模块，以便可以像一个整体一样进行前向传播。
        第一步：列表推导式：
        [
            DummyTransformerBlock(cfg)
            for _ in range(cfg["n_layers"])
        ]
        生成一个 长度为 n_layers 的列表，列表里的每一项是一个 DummyTransformerBlock 实例。假设 n_layers = 3，那么这个表达式结果是：
        [
            DummyTransformerBlock(cfg),
            DummyTransformerBlock(cfg),
            DummyTransformerBlock(cfg)
        ]
        第二步：星号展开 *[]：
        Python 中 * 是用来“解包列表”的。把列表拆成多个参数，传给 nn.Sequential。等价于
        self.trf_blocks = nn.Sequential(
            DummyTransformerBlock(cfg),
            DummyTransformerBlock(cfg),
            DummyTransformerBlock(cfg)
        )
        第三步：传入 nn.Sequential：
        self.trf_blocks = nn.Sequential(模块1, 模块2, 模块3, ...)。nn.Sequential 是 PyTorch 提供的一个容器，用来按顺序执行多个模块（layer）output = 模块3(模块2(模块1(input)))。只需要调用一次 .forward()，它就会按顺序执行这些子模块。
        """
        self.trf_blocks = nn.Sequential(
            *[
                DummyTransformerBlock(cfg)
                for _ in range(cfg["n_layers"])
            ]
        )

        #   层归一化，提升模型稳定性。
        self.final_norm = DummyLayerNorm(cfg["emb_dim"])

        #   线性输出层
        #   将每个位置的输出向量映射到词表大小的分布（用于预测下一个词元）。
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    """
    前向传播
    输入的形状为 (batch_size, seq_len)，batch_size 表示一个训练样本的个数，seq_len 表示一个训练样本的长度。如：
    txt1 = "Every effort moves you"
    txt2 = "Every day holds a"
    batch_size等于2，seq_len等于4
    """
    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        # print("in_idx.shape:", in_idx.shape)

        #   输入词元的词元嵌入
        #   查表获取每个词元的嵌入向量，形状变为 [batch_size, seq_len, emb_dim]。
        tok_embeds = self.tok_emb(in_idx)
        # print(f"tok_embeds.shape:", tok_embeds.shape)     #   tok_embeds.shape: torch.Size([2, 4, 768])

        #   输入词元的位置嵌入
        #   生成序列中每个位置的索引，并获取对应的位置嵌入（[seq_len, emb_dim]）。
        """
        GPT 模型是 基于 Transformer 架构的，而 Transformer 本身对输入的顺序是“感知不到的”。所以我们必须在输入中加上“位置信息”，这就是 位置嵌入（Positional Embedding） 的作用。
        输入的形状为 [seq_len]，输出的形状为 [seq_len, emb_dim]。加上 device=in_idx.device的目的保证位置张量和输入张量在 相同的设备（CPU 或 GPU） 上，防止报错。
        """
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        # print(f"pos_embeds.shape:", pos_embeds.shape)     #   pos_embeds.shape: torch.Size([4, 768])

        #   输入词元的位置嵌入和词元嵌入相加
        """
        用了PyTorch 中 广播机制（broadcasting），先对两个张量从右往左对齐维度。
        [batch_size, seq_len, emb_dim]，tok_embeds.shape: torch.Size([2, 4, 768])
        [seq_len, emb_dim]，pos_embeds.shape: torch.Size([4, 768])
        PyTorch 会把 pos_embeds 自动扩展为[batch_size, seq_len, emb_dim]，即torch.Size([1, 4, 768])
        二者形状相同，可以逐元素相加，结果为[batch_size, seq_len, emb_dim]
        告诉模型：“这个词是什么” + “它在什么位置”。
        """
        x = tok_embeds + pos_embeds

        #   输入词元的dropout隐藏单元被丢弃率
        x = self.drop_emb(x)

        #   输入词元的多个Transformer块处理
        #   目前是空壳，因为没有实现Transformer块。
        x = self.trf_blocks(x)

        #   输入词元的层归一化
        x = self.final_norm(x)

        #   输入词元的线性输出层
        #   输出是每个位置预测下一个词元的分布，形状为 [batch_size, seq_len, vocab_size]。
        logits = self.out_head(x)
        return logits

"""
Transformer块
"""
class DummyTransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()

    def forward(self, x):
        return x
    
"""
层归一化
提高神经网络训练的稳定性和效率
"""
class DummyLayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()

    def forward(self, x):
        return x
    

if  __name__ == "__main__":
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

    # 打印张量的形状和长度
    # Tensor 0: [6109, 3626, 6100, 345], length = 4
    # Tensor 1: [6109, 1110, 6622, 257], length = 4
    # for i, t in enumerate(batch):
    #     print(f"Tensor {i}: {t.tolist()}, length = {t.size(0)}")

    """torch.stack 这个函数名中的 "stack" 是英文单词，意思是 “堆叠”、“叠放”，它的功能也正如其名：将多个张量沿着一个新维度“堆叠”在一起，生成一个新的更高维的张量。
    注意：这里我犯了一个错误，没有将stack的结果保存到batch中，导致后面的forward操作出错了
    batch_size, seq_len = in_idx.shape
    AttributeError: 'list' object has no attribute 'shape'
    """
    batch = torch.stack(batch, dim=0)

    # 第一行对应第一段文本，第二行对应第二段文本
    # tensor([[6109, 3626, 6100,  345],
    #     [6109, 1110, 6622,  257]])
    print(batch)

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

    torch.manual_seed(123)

    """model = DummyGPTModel(GPT_CONFIG_124M)
        │
        └─► 自动调用 __init__(cfg)
               ├─► super().__init__()
               ├─► 初始化 tok_emb、pos_emb、drop_emb ...
               └─► 最后返回完整模型对象"""
    model = DummyGPTModel(GPT_CONFIG_124M)
    logits = model(batch)    
    
    print("Output shape:", logits.shape)    #   Output shape: torch.Size([2, 4, 50257])
    """    tensor([[[-0.9289,  0.2748, -0.7557,  ..., -1.6070,  0.2702, -0.5888],
            [-0.4476,  0.1726,  0.5354,  ..., -0.3932,  1.5285,  0.8557],
            [ 0.5680,  1.6053, -0.2155,  ...,  1.1624,  0.1380,  0.7425],
            [ 0.0447,  2.4787, -0.8843,  ...,  1.3219, -0.0864, -0.5856]],

            [[-1.5474, -0.0542, -1.0571,  ..., -1.8061, -0.4494, -0.6747],
            [-0.8422,  0.8243, -0.1098,  ..., -0.1434,  0.2079,  1.2046],
            [ 0.1355,  1.1858, -0.1453,  ...,  0.0869, -0.1590,  0.1552],
            [ 0.1666, -0.8138,  0.2307,  ...,  2.5035, -0.3055, -0.3083]]],
        grad_fn=<UnsafeViewBackward0>)"""
    print(logits)