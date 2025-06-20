import torch

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

## 第一步：随机生成权重矩阵 W_query、W_key 和 W_value
torch.manual_seed(123)  # 固定设置 PyTorch 的随机数种子（seed）为 123，以确保后续生成的随机数是“可复现的”。
W_query = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
W_key = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
W_value = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)

print(W_query.shape)    # 输出形状为：3 * 2

# 用 torch.rand(...) 分别生成了三个 独立的随机张量，虽然它们维度一样、随机种子一样，但它们的生成调用是三次独立的随机操作，每次都会生成不同的随机值。
# tensor([[0.2961, 0.5166],
#         [0.2517, 0.6886],
#         [0.0740, 0.8665]])
print(W_query)
# tensor([[0.1366, 0.1025],
#         [0.1841, 0.7264],
#         [0.3153, 0.6871]]) 
print(W_key)
# tensor([[0.0756, 0.1966],
#         [0.3164, 0.4017],
#         [0.1186, 0.8274]])
print(W_value)    

## 第二步：计算查询、键和值的向量
# 计算第二个词元的查询、键和值的向量
query_2 = x_2 @ W_query # query_2形状为：2，也就是1 * 3的形状 乘以 3 * 2的形状
key_2 = x_2 @ W_key
value_2 = x_2 @ W_value

print(query_2) # tensor([0.4306, 1.4551])

# 计算所有词元的键和值的向量
keys = inputs @ W_key
values = inputs @ W_value
print("keys.shape:",keys.shape) # 输出形状为：6 * 2
print("values.shape:",values.shape) # 输出形状为：6 * 2

## 第三步：计算注意力分数
# 单个词元的注意力分数
keys_2 = keys[1]
print(keys_2)   # tensor([0.4433, 1.1419])
attn_score_22 = query_2.dot(keys_2) # 点积 tensor([0.4433, 1.1419]) * tensor([0.4306, 1.4551])
print(attn_score_22)    # tensor(1.8524)

# 计算所有注意力分数
# query_2是第2个词元的查询向量，形状为2（相当于 (1, 2) 向量），而keys是所有词元的键向量，形状为6 * 2（也就是6个键向量，每个是2维的）
# keys的所有词元的键向量转置后，形状为2 * 6，所以结果形状为1 * 6，也就是1个查询向量和6个键向量之间的注意力分数。
# 注意：必须用转置 keys.T，因为矩阵乘法需要维度对齐，才能计算所有注意力分数。
attn_scores_2 = query_2 @  keys.T
print(attn_scores_2)    # tensor([1.2705, 1.8524, 1.8111, 1.0795, 0.5577, 1.5440])

## 第四步：计算注意力权重
d_k = keys.shape[-1]    # shape[-1] 是 Python 和 PyTorch 中的一种“倒数索引”写法，意思是 “取张量最后一个维度的大小”。也就是2
print(d_k)
# 除以根号d_k是为了分布更加稳定和防止 softmax 过尖锐（某些 token 被关注得太多，而其他 token 几乎被忽略）。结果为：tensor([0.1500, 0.2264, 0.2199, 0.1311, 0.0906, 0.1820])
# 没有除以根号d_k，softmax 过尖锐（某些 token 被关注得太多，而其他 token 几乎被忽略）。结果为：tensor([0.1401, 0.2507, 0.2406, 0.1157, 0.0687, 0.1842])
# attn_weights_2 = torch.softmax(attn_scores_2, dim=-1) 
attn_weights_2 = torch.softmax(attn_scores_2 / d_k ** 0.5, dim=-1)  
print(attn_weights_2)

## 第五步：计算上下文向量
context_vec_2 = attn_weights_2 @ values # 注意力向量乘以值向量进行加权求和
print(context_vec_2)    # tensor([0.3061, 0.8210])