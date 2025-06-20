import torch

inputs = torch.tensor([[0.43, 0.15, 0.89],
                       [0.55, 0.87, 0.66],
                       [0.57, 0.85, 0.64],
                       [0.22, 0.58, 0.33],
                       [0.77, 0.25, 0.10],
                       [0.05, 0.80, 0.55]])

query = inputs[1]   # 获取第 1 个向量

# print(query)
# print(inputs.shape) # 输出形状为：6 * 3

# inputs.shape[0] 取的是第 0 维的大小，也就是 6，表示有 6 个向量（通常代表 6 个 token 或样本）。
# torch.empty(6) 创建一个长度为 6 的 一维张量，只是申请内存空间，里面的值是未初始化的随机数（注意：不是 0）。因为我知道后面会逐个赋值，不需要初始化为 0，所以 empty() 会更高效一点。
attn_scores_2 = torch.empty(inputs.shape[0])
# print(attn_scores_2)    # 值是未初始化的随机数 如：tensor([-7.2621e+16,  2.1019e-42,  0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00])

for i, x_i in enumerate(inputs):
    attn_scores_2[i] = torch.dot(query, x_i)    # torch.dot() 函数用于计算两个张量的点积。也就是第2个词元与其他所有输入词元的点积。

"""
输出注意力分数的目的是度量 query 与 inputs 中每个向量之间的相似度。
点积越大，两个元素之间的相似度和注意力分数就越高。也就是反映绝对相似度。
"""
print(attn_scores_2)    # 输出注意力分数，tensor([0.9544, 1.4950, 1.4754, 0.8434, 0.7070, 1.0865])

# 用简单方法实现归一化
attn_scores_2_tmp = attn_scores_2/attn_scores_2.sum()  
print("Attention weights:", attn_scores_2_tmp) # 输出注意力分数归一化后的结果，tensor([0.1455, 0.2278, 0.2249, 0.1285, 0.1077, 0.1656])
print("Sum:",attn_scores_2_tmp.sum())   # 输出和，tensor(1.0000)

def softmax_naive(x):
    """
    用softmax函数实现归一化注意力分数。在处理大输入时，可能会出现数值溢出的情况，因此需要使用torch.softmax()函数来处理。
    反映相对重要性（概率分布）。权重越高，该元素对当前位置的贡献越大。归一化不改变元素间的相对顺序。
    :param x: 输入向量
    :return: softmax后的向量
    """
    return torch.exp(x)/torch.exp(x).sum(dim=0)

attn_scores_2_naive = softmax_naive(attn_scores_2)
print("Attention weights:", attn_scores_2_naive) # 输出注意力分数归一化后的结果，tensor([0.1385, 0.2379, 0.2333, 0.1240, 0.1082, 0.1581])
print("Sum:",attn_scores_2_naive.sum())   # 输出和，tensor(1.)

# 使用torch.softmax()函数
attn_weights_2 = torch.softmax(attn_scores_2, dim=0)    # dim=0 表示在第 0 维（也就是张量 attn_scores_2 是 1维 的（形状为 [6]）这个向量本身）上做 softmax。
print("Attention weights:", attn_weights_2) # 输出注意力分数归一化后的结果，tensor([0.1385, 0.2379, 0.2333, 0.1240, 0.1082, 0.1581])
print("Sum:",attn_weights_2.sum())   # 输出和，tensor(1.)

# 计算上下文向量
# 通过将每个输入向量乘以对应的注意力权重，然后求和得到上下文向量。
context_vec_2 = torch.zeros(query.shape)    # 初始化形状，将第2个输入词元作为查询向量，结果为tensor([0., 0., 0.])
for i, x_i in enumerate(inputs):
    # 0: 0.13854756951332092 * tensor([0.4300, 0.1500, 0.8900])
    # tensor([0.0596, 0.0208, 0.1233])
    # 1: 0.2378913015127182 * tensor([0.5500, 0.8700, 0.6600])
    # tensor([0.1308, 0.2070, 0.1570])
    # 2: 0.23327402770519257 * tensor([0.5700, 0.8500, 0.6400])
    # tensor([0.1330, 0.1983, 0.1493])
    # 3: 0.12399158626794815 * tensor([0.2200, 0.5800, 0.3300])
    # tensor([0.0273, 0.0719, 0.0409])
    # 4: 0.10818186402320862 * tensor([0.7700, 0.2500, 0.1000])
    # tensor([0.0833, 0.0270, 0.0108])
    # 5: 0.15811361372470856 * tensor([0.0500, 0.8000, 0.5500])
    # tensor([0.0079, 0.1265, 0.0870])
    print(f"{i}: {attn_weights_2[i]} * {x_i}")
    print(attn_weights_2[i] * x_i)
    context_vec_2 += attn_weights_2[i] * x_i
print(context_vec_2)       # tensor([0.4419, 0.6515, 0.5683])
