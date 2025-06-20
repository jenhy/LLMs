import torch
import torch.nn as nn

class CausalAttention(nn.Module):
    """
    使用 nn.Linear 层定义权重，支持偏置，通过qkv_bias参数控制是否使用偏置。
    """
    def __init__(self, d_in, d_out, context_length, dropout, qkv_bias=False):
        """
        d_in: 输入向量的维度
        d_out: 输出向量(Query/Key/Value)的维度
        context_length: 输入向量的长度（token 数量）
        dropout: 丢弃概率
        qkv_bias: 是否使用偏置
        """
        super().__init__()
        self.d_out = d_out
        # 用来生成Query（查询）、Key（键）、Value（值）向量的权重矩阵。
        # self.W_query.weight.shape是(3, 2)。nn.Linear 会将最后一个维度 d_in 自动映射为 d_out。代码层面可以看看 PyTorch 中 nn.Linear 的核心实现。
        # 下同
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)  
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)  
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

        # 在注意力权重上使用 dropout，以防过拟合。
        # self.dropout 是一个 nn.Dropout 层实例，也就是一个 PyTorch 模块（Module），它是对象，不是函数。
        # self.dropout 是一个类属性，类型是 nn.Dropout
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
        """
        self.register_buffer('mask', torch.triu(torch.ones(context_length, context_length), diagonal=1))
    def forward(self, x):
        """
        前向传播
        x: 输入张量,形状为(batch_size, num_tokens, d_in)
        """

        """
        b: 批次大小
        num_tokens: token 数
        d_in: 输入维度
        """
        b, num_tokens, d_in = x.shape

        # 1、计算 Query、Key、Value 向量
        # @ 是矩阵乘法。将每个输入向量投影到新的空间中。输出形状为(b, num_tokens, d_out)
        queries = self.W_query(x)
        keys = self.W_key(x)
        values = self.W_value(x)

        # 2、计算注意力分数
        # queries: (b, N, d_out)
        # keys.transpose(1, 2): (b, d_out, N)，也就是将维度1和2进行转置，以匹配矩阵乘法的要求。
        # 相乘后 attn_score: (b, N, N)
        attn_score = queries @ keys.transpose(1, 2)

        """
        attn_score:是 (b, N, N) 的注意力分数矩阵（batch × token × token）。每个元素表示第 i 个 token 对第 j 个 token 的注意力强度（打分）。
        self.mask：这个 mask 是我们在 __init__ 时定义的：self.register_buffer('mask', torch.triu(torch.ones(context_length, context_length), diagonal=1))。意思是：每一行只能“看”自己以及左边的 token，不能看右边的 token（未来）。
        self.mask.bool()[:num_tokens, :num_tokens]：将 mask 转成布尔类型（True 表示要被 mask 的位置）。然后根据当前输入序列长度 num_tokens 做裁剪，表示从 mask 中裁剪出当前实际 token 数量的子矩阵，用作注意力遮挡区域的布尔掩码。保证对齐实际 token 数量。
        masked_fill_(mask, value)：这是 PyTorch 的就地操作（带 _），意思是：对于 mask 为 True 的位置，将 attn_score 中对应元素赋值为 -inf。attn_score[mask == True] = -inf
        总结来说：用布尔掩码把未来的 token 注意力打分设置为 -inf，确保 softmax 后对应的注意力权重为 0，实现“只能看过去，不能看未来”的因果自注意力机制（causal self-attention）。
        """
        attn_score.masked_fill_(self.mask.bool()[:num_tokens, :num_tokens], -torch.inf)
        
        # 3、缩放（Scaling）+ Softmax 归一化
        # shape[-1] 是 Python 和 PyTorch 中的一种“倒数索引”写法，意思是 “取张量最后一个维度的大小”。也就是d_out
        # attn_weights: (b, N, N)，注意力权重矩阵。
        attn_weights = torch.softmax(attn_score / keys.shape[-1] ** 0.5, dim=-1)

        #  PyTorch 模块都实现了 __call__() 方法，所以你可以像函数一样调用它。实际上等价于attn_weights = self.dropout.__call__(attn_weights)
        attn_weights = self.dropout(attn_weights)

        # 4、计算上下文向量（Context Vector）
        # attn_weight: (b, N, N),values: (b, N, d_out),输出形状为(b, N, d_out)
        context = attn_weights @ values
        return context


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
    d_in = inputs.shape[1] # 输入嵌入维度3
    d_out = 2 # 输出嵌入维度2
    context_length = batch.shape[1] # 取第一维的大小，也就是序列长度（token 数量6）
    ca = CausalAttention(d_in, d_out, context_length, 0.0)
    context_vecs = ca(batch) # 相当于context_vecs = ca.__call__(batch)。所有继承自 nn.Module 的模块，都实现了 __call__() 方法。内部会自动调用你写的 forward() 方法
    # torch.Size([2, 6, 2])
    print("context_vecs.shape:", context_vecs.shape)
    """
    tensor([[[-0.5716, -0.1203],
         [-0.3553, -0.1572],
         [-0.2804, -0.1623],
         [-0.2029, -0.1659],
         [-0.1941, -0.0703],
         [-0.1565, -0.1281]],

        [[-0.5716, -0.1203],
         [-0.3553, -0.1572],
         [-0.2804, -0.1623],
         [-0.2029, -0.1659],
         [-0.1941, -0.0703],
         [-0.1565, -0.1281]]], grad_fn=<UnsafeViewBackward0>)
    """
    print("context_vecs:", context_vecs)