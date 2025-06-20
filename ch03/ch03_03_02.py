import torch
import time

inputs = torch.tensor([[0.43, 0.15, 0.89],
                       [0.55, 0.87, 0.66],
                       [0.57, 0.85, 0.64],
                       [0.22, 0.58, 0.33],
                       [0.77, 0.25, 0.10],
                       [0.05, 0.80, 0.55]])

# torch.empty(6) 创建一个长度为 6 * 6 的 二维张量，只是申请内存空间，里面的值是未初始化的随机数（注意：不是 0）。因为我知道后面会逐个赋值，不需要初始化为 0，所以 empty() 会更高效一点。
attn_scores = torch.empty(6, 6)
# print(attn_scores)    # 值是未初始化的随机数

start_time = time.time()

### 第一步：计算注意力分数
## 使用for循环
# 把 start_time（是一个 float 秒数，自 1970 年起）转换为本地时间的时间结构体（struct_time），例如：年月日时分秒。
# strftime = “string format time”，把时间结构体格式化为字符串。"%Y-%m-%d %H:%M:%S" 表示格式为：年-月-日 小时:分钟:秒（24 小时制），如 2025-05-10 17:05:38。
# start_time % 1 取的是秒的小数部分，例如 1684392843.345678 % 1 = 0.345678
# :.6f 表示保留 6 位小数（即微秒精度）所以输出会是 0.345678 seconds，表示这个时间点的“微秒部分”。
# 拼接起来打印：Start time: 2025-05-10 17:05:38 0.713640 seconds
print("Start time:", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time)), f"{(start_time % 1):.6f} seconds")
for i, x_i in enumerate(inputs):
    for j, x_j in enumerate(inputs):
        attn_scores[i, j] = torch.dot(x_i, x_j)
end_time = time.time()
print("End time:", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(end_time)), f"{(end_time % 1):.6f} seconds")

# 输出注意力分数:
# tensor([[0.9995, 0.9544, 0.9422, 0.4753, 0.4576, 0.6310],
#         [0.9544, 1.4950, 1.4754, 0.8434, 0.7070, 1.0865],
#         [0.9422, 1.4754, 1.4570, 0.8296, 0.7154, 1.0605],
#         [0.4753, 0.8434, 0.8296, 0.4937, 0.3474, 0.6565],
#         [0.4576, 0.7070, 0.7154, 0.3474, 0.6654, 0.2935],
#         [0.6310, 1.0865, 1.0605, 0.6565, 0.2935, 0.9450]])
print(attn_scores)    

## 使用矩阵乘法
print("Start time:", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time)), f"{(start_time % 1):.6f} seconds")
attn_scores = inputs @ inputs.T # 张量的矩阵乘法，等价于 torch.matmul(inputs, inputs.T)。inputs的形状是（6 * 3），inputs.T表示对 inputs 做转置，形状是（3 * 6），inputs @ inputs.T 是矩阵乘法，结果形状为 (6, 6)。
print("End time:", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(end_time)), f"{(end_time % 1):.6f} seconds")
# 输出注意力分数:
# tensor([[0.9995, 0.9544, 0.9422, 0.4753, 0.4576, 0.6310],
#         [0.9544, 1.4950, 1.4754, 0.8434, 0.7070, 1.0865],
#         [0.9422, 1.4754, 1.4570, 0.8296, 0.7154, 1.0605],
#         [0.4753, 0.8434, 0.8296, 0.4937, 0.3474, 0.6565],
#         [0.4576, 0.7070, 0.7154, 0.3474, 0.6654, 0.2935],
#         [0.6310, 1.0865, 1.0605, 0.6565, 0.2935, 0.9450]])
print(attn_scores)    

### 第二步：计算注意力权重
# 使用torch.softmax()函数
attn_weights_2 = torch.softmax(attn_scores, dim=-1)    # dim=-1 表示在最后一维，对形状（行，列）来说，也就是对列进行归一化处理，使得每一行的总和为 1。

# 输出注意力分数归一化后的结果
# tensor([[0.2098, 0.2006, 0.1981, 0.1242, 0.1220, 0.1452],
#         [0.1385, 0.2379, 0.2333, 0.1240, 0.1082, 0.1581],
#         [0.1390, 0.2369, 0.2326, 0.1242, 0.1108, 0.1565],
#         [0.1435, 0.2074, 0.2046, 0.1462, 0.1263, 0.1720],
#         [0.1526, 0.1958, 0.1975, 0.1367, 0.1879, 0.1295],
#         [0.1385, 0.2184, 0.2128, 0.1420, 0.0988, 0.1896]])
print("Attention weights:", attn_weights_2) 
print("Sum:",attn_weights_2.sum(dim=-1))   # 输出Sum: tensor([1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000])

# 测试和是否为1
row_sum = sum([0.1385, 0.2379, 0.2333, 0.1240, 0.1082, 0.1581])
print("Row sum:", row_sum)

### 第三步：计算上下文向量
# 计算所有上下文向量
# tensor([[0.4421, 0.5931, 0.5790],
#         [0.4419, 0.6515, 0.5683],
#         [0.4431, 0.6496, 0.5671],
#         [0.4304, 0.6298, 0.5510],
#         [0.4671, 0.5910, 0.5266],
#         [0.4177, 0.6503, 0.5645]])
all_context_vec = attn_weights_2 @ inputs
print(all_context_vec)
