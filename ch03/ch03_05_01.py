import torch
import torch.nn as nn   # 构建神经网络netural network的模块


class SelfAttentionV2(nn.Module):
    """
    使用 nn.Linear 层定义权重，支持偏置，通过qkv_bias参数控制是否使用偏置。
    """
    def __init__(self, d_in, d_out, qkv_bias=False):
        """
        d_in: 输入向量的维度
        d_out: 输出向量(Query/Key/Value)的维度
        qkv_bias: 是否使用偏置
        """
        super().__init__()
        # 用来生成Query（查询）、Key（键）、Value（值）向量的权重矩阵。
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)  
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

    def forward(self, x):
        """
        前向传播
        x: 输入张量,形状为(N,d_in)
        """

        # 1、计算 Query、Key、Value 向量
        # @ 是矩阵乘法。将每个输入向量投影到新的空间中。输出形状为(N,d_out)
        queries = self.W_query(x)
        keys = self.W_key(x)
        values = self.W_value(x)

        # 2、计算注意力分数
        # queries: (N, d_out),keys.T: (d_out, N),输出形状为(N, N)
        attn_score = queries @ keys.T

        # 3、缩放（Scaling）+ Softmax 归一化
        # shape[-1] 是 Python 和 PyTorch 中的一种“倒数索引”写法，意思是 “取张量最后一个维度的大小”。也就是d_out
        # attn_weight: (N, N)，注意力权重矩阵。
        attn_weight = torch.softmax(attn_score / keys.shape[-1] ** 0.5, dim=-1)

        # 4、计算上下文向量（Context Vector）
        # attn_weight: (N, N),values: (N, d_out),输出形状为(N,d_out)
        context = attn_weight @ values
        return context
    
if __name__ == '__main__':
    inputs = torch.tensor([[0.43, 0.15, 0.89],
                       [0.55, 0.87, 0.66],
                       [0.57, 0.85, 0.64],
                       [0.22, 0.58, 0.33],
                       [0.77, 0.25, 0.10],
                       [0.05, 0.80, 0.55]])
    x_2 = inputs[1] # x_2形状为：3
    print(x_2.shape)
    d_in = inputs.shape[1] # 输入嵌入维度3
    d_out = 2 # 输出嵌入维度2

    ## 第一步：使用softmax函数计算注意力权重
    torch.manual_seed(789)
    sa_v2 = SelfAttentionV2(d_in, d_out)
    queries = sa_v2.W_query(inputs)
    keys = sa_v2.W_key(inputs)
    values = sa_v2.W_value(inputs)
    attn_scores = queries @ keys.T
    attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
    # tensor([[0.1921, 0.1646, 0.1652, 0.1550, 0.1721, 0.1510],
    #     [0.2041, 0.1659, 0.1662, 0.1496, 0.1665, 0.1477],
    #     [0.2036, 0.1659, 0.1662, 0.1498, 0.1664, 0.1480],
    #     [0.1869, 0.1667, 0.1668, 0.1571, 0.1661, 0.1564],
    #     [0.1830, 0.1669, 0.1670, 0.1588, 0.1658, 0.1585],
    #     [0.1935, 0.1663, 0.1666, 0.1542, 0.1666, 0.1529]],
    #    grad_fn=<SoftmaxBackward0>)
    print(attn_weights)

    ## 第二步：将对角线以上元素置为0的掩码
    # 这行代码的意思是：取 attn_scores 的第 0 个维度大小，通常这个维度代表序列的长度（token 数量）。举例：如果 attn_scores 是 (6, 6)，那么 context_length = 6。
    context_length = attn_scores.shape[0]
    # 先创建一个 context_length × context_length 的全 1 矩阵。
    # tril 是 “lower triangular” 的意思，它保留矩阵的下三角部分（包括对角线），其余部分设为 0。
    # tensor([[1., 0., 0., 0., 0., 0.],
    #     [1., 1., 0., 0., 0., 0.],
    #     [1., 1., 1., 0., 0., 0.],
    #     [1., 1., 1., 1., 0., 0.],
    #     [1., 1., 1., 1., 1., 0.],
    #     [1., 1., 1., 1., 1., 1.]])
    mask_simple = torch.tril(torch.ones(context_length, context_length))
    print(mask_simple)

    masked_simple = attn_weights * mask_simple  # 逐元素相乘,而不是矩阵乘法
    # tensor([[0.1921, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
    #     [0.2041, 0.1659, 0.0000, 0.0000, 0.0000, 0.0000],
    #     [0.2036, 0.1659, 0.1662, 0.0000, 0.0000, 0.0000],
    #     [0.1869, 0.1667, 0.1668, 0.1571, 0.0000, 0.0000],
    #     [0.1830, 0.1669, 0.1670, 0.1588, 0.1658, 0.0000],
    #     [0.1935, 0.1663, 0.1666, 0.1542, 0.1666, 0.1529]],
    #    grad_fn=<MulBackward0>)
    print(masked_simple)

    ## 第三步：重新归一化注意力权重，使得每一行的总和再次为1
    row_sums = masked_simple.sum(dim=-1, keepdim=True)  # 按最后一个维度求和（即行内求和）。dim 指的是要“消除”（或者说压缩、聚合）哪个维度。跟我理解的相反，需要注意。keepdim=True：保持原来的张量维度（为了广播除法）。
    masked_simple_norm = masked_simple / row_sums
    # tensor([[1.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
    #     [0.5517, 0.4483, 0.0000, 0.0000, 0.0000, 0.0000],
    #     [0.3800, 0.3097, 0.3103, 0.0000, 0.0000, 0.0000],
    #     [0.2758, 0.2460, 0.2462, 0.2319, 0.0000, 0.0000],
    #     [0.2175, 0.1983, 0.1984, 0.1888, 0.1971, 0.0000],
    #     [0.1935, 0.1663, 0.1666, 0.1542, 0.1666, 0.1529]],
    #    grad_fn=<DivBackward0>)
    print(masked_simple_norm)

    ## 利用softmax函数的数学特性进行改进优化。
    # torch.ones(context_length, context_length) 生成一个全是 1 的二维矩阵，形状为 (N, N)。
    # torch.triu(..., diagonal=1) 从这个矩阵中取出主对角线以上的部分（即严格的上三角），对角线和下三角都是 0。
    # 最终生成的 mask 是一个掩码矩阵，形状仍是 (N, N)
    # mask如下：
    # tensor([[0., 1., 1., 1., 1., 1.],
    #     [0., 0., 1., 1., 1., 1.],
    #     [0., 0., 0., 1., 1., 1.],
    #     [0., 0., 0., 0., 1., 1.],
    #     [0., 0., 0., 0., 0., 1.],
    #     [0., 0., 0., 0., 0., 0.]])
    mask = torch.triu(torch.ones(context_length, context_length), diagonal=1)

    # mask.bool() 把掩码转成布尔类型（True 表示要屏蔽）。
    # .masked_fill(..., -torch.inf)：把 attn_scores 中那些位置为 True 的地方，填上负无穷大(-inf)。
    # 最终结果：对角线及其下方的部分保留原始值；对角线上方的部分（即“未来”）全部变为 -inf。
    masked = attn_scores.masked_fill(mask.bool(), -torch.inf)

    # tensor([[0.2899,   -inf,   -inf,   -inf,   -inf,   -inf],
    #     [0.4656, 0.1723,   -inf,   -inf,   -inf,   -inf],
    #     [0.4594, 0.1703, 0.1731,   -inf,   -inf,   -inf],
    #     [0.2642, 0.1024, 0.1036, 0.0186,   -inf,   -inf],
    #     [0.2183, 0.0874, 0.0882, 0.0177, 0.0786,   -inf],
    #     [0.3408, 0.1270, 0.1290, 0.0198, 0.1290, 0.0078]],
    #    grad_fn=<MaskedFillBackward0>)
    print(masked)

    attn_weights = torch.softmax(masked / keys.shape[-1] ** 0.5, dim=1)
    # tensor([[1.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
    #     [0.5517, 0.4483, 0.0000, 0.0000, 0.0000, 0.0000],
    #     [0.3800, 0.3097, 0.3103, 0.0000, 0.0000, 0.0000],
    #     [0.2758, 0.2460, 0.2462, 0.2319, 0.0000, 0.0000],
    #     [0.2175, 0.1983, 0.1984, 0.1888, 0.1971, 0.0000],
    #     [0.1935, 0.1663, 0.1666, 0.1542, 0.1666, 0.1529]],
    #    grad_fn=<SoftmaxBackward0>)
    print(attn_weights)

    ## 第四步：计算上下文向量
    context_vec = attn_weights @ values
    # tensor([[-0.0872,  0.0286],
    #     [-0.0991,  0.0501],
    #     [-0.0999,  0.0633],
    #     [-0.0983,  0.0489],
    #     [-0.0514,  0.1098],
    #     [-0.0754,  0.0693]], grad_fn=<MmBackward0>)
    print(context_vec)