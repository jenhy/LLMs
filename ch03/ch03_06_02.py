import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert (d_out % num_heads == 0), \
        "d_out must be divisible by num_heads"  # 这句断言是为了确保能均匀地把输出向量分给每个注意力头，避免维度错误或模型结构不一致。

        self.d_out = d_out  # 输出嵌入维度，比如2
        self.num_heads = num_heads  # 多头注意力的头数，比如2

        """
        表示每个注意力头处理的子空间维度。
        总共有 num_heads 个注意力头（比如 8 个头）；
        整个输出向量的维度是 d_out（比如 512）；
        每个头会负责处理这 512 维中的一部分子空间；
        所以我们要把 d_out 平均分成 num_heads 份；
        为什么这样设计？
        并行处理：每个头学习不同的注意力模式。
        降维解耦：每个头工作在一个低维空间（而不是整个高维空间），有助于模型泛化。
        拼接后信息更丰富：多个注意力头拼接的信息更全面，再通过 out_proj 融合。
        """
        self.head_dim = d_out // num_heads
        # print(f"self.d_out: {self.d_out},self.num_heads:  {self.num_heads},self.head_dim: {self.head_dim}")

        # 上一个版本是每个头一个线性层；现在是把多个头融合在一起一次性映射。
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

        """
        使用一个线性变换层来组合头的输出。它接收这个拼接后的张量，再做一次全连接变换，把多头的信息融合起来，输出的维度仍然是 d_out，方便后续模块使用。
        第一个 d_out：表示输入维度，它是多个注意力头拼接后的维度：d_out = num_heads × head_dim。
        第二个 d_out：表示输出维度，是线性变换之后的输出维度（希望得到的最终输出维度）。
        """
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)
        """
        torch.ones(context_length, context_length),创建一个大小为 (context_length, context_length) 的张量，所有值为 1。表示每个 token 和其他 token 的注意力权重位置。
        torch.triu(..., diagonal=1),取矩阵的上三角部分（不包含对角线）,如：
        [[0, 1, 1, 1],
        [0, 0, 1, 1],
        [0, 0, 0, 1],
        [0, 0, 0, 0]]
        这就构成了掩码矩阵（mask）：用于遮蔽未来的位置（右边的 token）。
        register_buffer('mask', ...)。跟随模型自动转到 GPU/CPU上，也会随着模型保存/加载一起保留。将生成的mask注册为模块的一个“缓冲区”（非参数张量）：不会作为 model.parameters() 的一部分进行训练。但会在 model.to(device) 或 model.eval() 等操作时一并处理。
        总结来说：这行代码是生成一个上三角掩码矩阵，防止模型在自注意力计算中看到未来 token，确保信息流动方向是“从左到右”。
        注意：区分torch.triu和torch.tril的区别。torch.triu（上三角），目的是屏蔽未来的 token，而torch.tril（下三角），目的是所有可用的注意力位置都变成-inf
        """
        self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length), diagonal=1))


    def forward(self, x):
        b, num_tokens, d_in = x.shape

        # 形状:(b, num_tokens, d_out)
        queries = self.W_query(x)
        keys = self.W_key(x)
        values = self.W_value(x)

        """
        把输入向量分割改变成多个注意力头。即把原来“合并”在一起的 d_out 维度，重新分割成 num_heads 个小头，每个小头 head_dim 维。
        类比：
        想象你有一个 6×12 的拼图（12 是总维度），你想把它拆成 3 块拼图（num_heads = 3），每块是 6×4（head_dim = 4），这样每个头可以单独处理自己的小块信息，不会互相干扰。
        为什么必须这么做？因为多头注意力机制的核心思想就是：并行执行多个独立注意力计算。
        为什么叫.view()？：.view() 是一个 PyTorch 的方法，用于改变张量的形状。名字 “view” 源自“视图（view）”这个概念 —— 它返回的是一个新的张量视图（不是数据的复制），只是以不同的形状/不同的角度来“看待”原始数据。不同于.reshape()，.reshape() 更灵活，必要时会创建数据副本。
        """
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)

        # 形状:(b, num_tokens, num_heads, head_dim)转换或者叫转置为(b, num_heads, num_tokens, head_dim)
        # 目的：把 attention head 维度提前（从第 2 维调到第 1 维）是为了让每个头可以并行处理整个序列，而不是每个 token 自己处理自己的多个头。
        queries = queries.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)

        ## 计算每个头的注意力分数
        # 计算每个头的点积。即(b, num_heads, num_tokens, head_dim) @ (b, num_heads, head_dim, num_tokens) = (b, num_heads, num_tokens, num_tokens)
        attn_scores = queries @ keys.transpose(2, 3)
        # self.mask.bool()[:num_tokens, :num_tokens]：将 mask 转成布尔类型（True 表示要被 mask 的位置）。然后根据当前输入序列长度 num_tokens 做裁剪，表示从 mask 中裁剪出当前实际 token 数量的子矩阵，用作注意力遮挡区域的布尔掩码。保证对齐实际 token 数量。
        # mask_bool = mask_bool.to(attn_scores.device): 将 mask_bool 移动到与 attn_scores 同一设备上。
        # masked_fill_(mask, value)：这是 PyTorch 的就地操作（带 _），意思是：对于 mask 为 True 的位置，将 attn_score 中对应元素赋值为 -inf。attn_score[mask == True] = -inf
        # 总结来说：用布尔掩码把未来的 token 注意力打分设置为 -inf，确保 softmax 后对应的注意力权重为 0，实现“只能看过去，不能看未来”的因果自注意力机制（causal self-attention）。
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        mask_bool = mask_bool.to(attn_scores.device)
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        ## 计算注意力权重
        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
        # PyTorch 模块都实现了 __call__() 方法，所以你可以像函数一样调用它。实际上等价于attn_weights = self.dropout.__call__(attn_weights)
        # attn_weights形状：(b, num_heads, num_tokens, num_tokens)
        attn_weights = self.dropout(attn_weights)
        # print("attn_weights.shape:", attn_weights.shape)

        ## 计算上下文向量
        """
        attn_weights形状：(b, num_heads, num_tokens, num_tokens) @ 值向量形状(b, num_heads, num_tokens, head_dim) = (b, num_heads, num_tokens, head_dim)
        再将(b, num_heads, num_tokens, head_dim)的第1维调到第2维，结果为(b, num_tokens, num_heads, head_dim)
        再进行拼接，结果为(b, num_tokens, num_heads * head_dim)，即输出的维度为(b, num_tokens, d_out)，但这个拼接只是简单把多个头的向量“堆叠”在一起，这些信息之间还没有“混合”或“交互”。
        最后将拼接后的多头输出，通过一个线性层（全连接层）进行再映射融合，以形成模型最终的输出向量。类比：可以把多头注意力比作多个“专家”（多个头）各自给出建议（向量），这一步 self.out_proj 就像是一个最终的裁决者，综合各位专家的建议，做出一个统一决策。虽然执行self.out_proj之前的形状和执行self.out_proj之后的形状相同，但是执行self.out_proj后，语义已经完全发生了变化，这也是线性层的作用。

        为什么要换维度？为了准备将多头结果拼接。
        假设：
        num_heads = 2
        head_dim = 4 → d_out = 8
        如果你现在是 context_vec.shape = (b, 2, num_tokens, 4)，直接 view 成 (b, num_tokens, 8) 会在 head 维度上错位拼接。
        但如果你先转置为：(b, num_tokens, 2, 4) → 然后再 view → (b, num_tokens, 8)
        就会是正确的拼接顺序：把每个 token 的所有 head 的结果拼接到一起，形成最终的输出向量。
        """
        context_vec = (attn_weights @ values).transpose(1, 2)
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)
        return context_vec
    
if __name__ == '__main__':
    inputs = torch.tensor([[0.43, 0.15, 0.89],
                        [0.55, 0.87, 0.66],
                        [0.57, 0.85, 0.64],
                        [0.22, 0.58, 0.33],
                        [0.77, 0.25, 0.10],
                        [0.05, 0.80, 0.55]])

    """
    torch.stack 这个函数名中的 "stack" 是英文单词，意思是 “堆叠”、“叠放”，它的功能也正如其名：将多个张量沿着一个新维度“堆叠”在一起，生成一个新的更高维的张量。
    将两个形状为 (6, 3) 的张量沿着新维度（第 0 维）堆叠成一个形状为 (2, 6, 3) 的三维张量。
    第 0 维：表示有2个“输入样本集”(可以理解为 batch size = 2)
    第 1 维：每个样本集有 6 个样本
    第 2 维：每个样本有 3 个特征
    """
    batch = torch.stack((inputs, inputs), dim=0)
    # print(batch)
    d_out = 2 # 输出嵌入维度2

    torch.manual_seed(123)
    batch_size, context_length, d_in = batch.shape # torch.Size([2, 6, 3])
    mha = MultiHeadAttention(d_in, d_out, context_length, 0.0, num_heads=2)
    context_vecs = mha(batch)
    """
    tensor([[[0.3190, 0.4858],
         [0.2943, 0.3897],
         [0.2856, 0.3593],
         [0.2693, 0.3873],
         [0.2639, 0.3928],
         [0.2575, 0.4028]],

        [[0.3190, 0.4858],
         [0.2943, 0.3897],
         [0.2856, 0.3593],
         [0.2693, 0.3873],
         [0.2639, 0.3928],
         [0.2575, 0.4028]]], grad_fn=<ViewBackward0>)
    """
    print(context_vecs)
    print("context_vecs.shape:", context_vecs.shape)    # torch.Size([2, 6, 2])

    # 练习3.3
    batch = torch.rand(2, 1024, 768)
    d_in, d_out = 768, 768
    context_length = 1024
    mha = MultiHeadAttention(d_in, d_out, context_length, 0.0, num_heads=12)
    context_vecs = mha(batch)
    """tensor([[[ 0.0671,  0.1007, -0.1969,  ...,  0.0968,  0.1423,  0.2266],
         [-0.0202,  0.0717, -0.1709,  ...,  0.0177,  0.2170,  0.1708],
         [ 0.0512,  0.0730, -0.1553,  ...,  0.0404,  0.1971,  0.1639],
         ...,
         [ 0.0426,  0.1550, -0.1796,  ...,  0.0171,  0.1010,  0.1204],
         [ 0.0428,  0.1551, -0.1796,  ...,  0.0169,  0.1011,  0.1202],
         [ 0.0426,  0.1552, -0.1795,  ...,  0.0171,  0.1012,  0.1201]],

        [[ 0.1522,  0.0838, -0.3283,  ..., -0.0735,  0.0821,  0.1365],
         [ 0.0260,  0.0861, -0.2724,  ..., -0.0053,  0.0732,  0.1217],
         [ 0.0608,  0.1271, -0.2851,  ...,  0.0191,  0.0738,  0.1428],
         ...,
         [ 0.0450,  0.1514, -0.1875,  ...,  0.0142,  0.0933,  0.1169],
         [ 0.0445,  0.1510, -0.1875,  ...,  0.0142,  0.0934,  0.1170],
         [ 0.0444,  0.1511, -0.1875,  ...,  0.0141,  0.0933,  0.1170]]],
       grad_fn=<ViewBackward0>)"""
    print(context_vecs)
    print("context_vecs.shape:", context_vecs.shape)    # torch.Size([2, 1024, 768])
