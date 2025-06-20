import torch
import matplotlib.pyplot as plt

vocab = {
    "closer": 0,
    "every": 1,
    "effort": 2,
    "forward": 3,
    "inches": 4,
    "moves": 5,
    "pizza": 6,
    "toward": 7,
    "you": 8
}

inverse_vocab = {v: k for k, v in vocab.items()}
print(inverse_vocab)

next_token_logits = torch.tensor([4.51, 0.89, -1.90, 6.75, 1.63, -1.62, -1.89, 6.28, 1.79])

probas = torch.softmax(next_token_logits, dim=0)
torch.set_printoptions(sci_mode=False)  # 设置打印选项，禁用科学计数法

"""
输出概率分布：
Probas: tensor([    0.0609,     0.0016,     0.0001,     0.5721,     0.0034,     0.0001,
            0.0001,     0.3576,     0.0040])
"""
print("Probas:", probas)

next_token_id = torch.argmax(probas).item()
print("Next token id:", next_token_id)  # Next token id: 3
print(inverse_vocab[next_token_id]) # forward


## 使用概率采样的multinomial函数替换argmax函数
torch.manual_seed(123)

"""
torch.multinomial 是从给定的概率分布中进行采样的函数。
probas 是一个一维的概率向量（比如 [0.1, 0.3, 0.6]），表示每个 token 的概率。
num_samples=1 表示我们要从中随机采样一个元素的索引（也就是一个 token 的 id）。
item()是把只包含一个元素的 Tensor 转换为 Python 标量。
"""
next_token_id = torch.multinomial(probas, num_samples=1).item()
print("Next token id:", next_token_id)
print(inverse_vocab[next_token_id])

def print_sampled_tokens(probas):
    """
    从给定的概率分布 probas 中采样 1000 次 token id，统计每个 token 被采样的频率，并打印出来（用 token 名称表示）。
    """
    torch.manual_seed(123)

    """
    执行 1000 次采样：
    每次从 probas（一个概率向量）中随机选出一个 token id。
    torch.multinomial(probas, num_samples=1) → 返回一个只包含 1 个元素的 tensor，如 tensor([2])
    .item() → 把 tensor 转换成 Python 整数，如 2
    所以最终 sample 是一个长度为 1000 的 list
    """
    sample = [torch.multinomial(probas, num_samples=1).item() for i in range(1_000)]

    # 把 sample 列表转换成 tensor，统计每个 token id 出现的次数。
    # 如果有 3 个 token（id 为 0、1、2）：[2, 1, 2, 0, 2, 2, 1, 0],结果为tensor([2, 2, 4])  # 表示 token 0 出现 2 次，1 出现 2 次，2 出现 4 次
    sample_ids = torch.bincount(torch.tensor(sample))
    for i, freq in enumerate(sample_ids):
        print(f"{freq} * {inverse_vocab[i]}")

print_sampled_tokens(probas)

## 温度缩放
def softmax_with_temperature(logits, temperature):
    # logits: [batch_size, vocab_size]
    # temperature: float
    # 返回一个张量，表示在给定温度下，每个 logit 的概率分布。
    probas = torch.softmax(logits / temperature, dim=0)
    return probas

## 对不同温度值分别打印采样频率
temperatures = [1, 0.1, 5]
for T in temperatures:
    print(f"\n Sample with temperature {T}")
    probas = softmax_with_temperature(next_token_logits, T)
    print_sampled_tokens(probas)

## 绘制不同温度值的概率分布
temperatures = [1, 0.1, 5]  # 选择了三个温度值：标准（1）、低温（0.1）、高温（5），用于比较。
scaled_probas = [softmax_with_temperature(next_token_logits, T) for T in temperatures]  # 计算不同温度下的 softmax 概率分布
print("scaled_probas:", scaled_probas)
x = torch.arange(len(vocab))    #  x 轴位置（每个 token 的编号）
bar_width = 0.15    # 每组柱子的宽度
fig, ax = plt.subplots(figsize=(5, 3))  # 创建图表

"""
绘制每种温度下的柱状图
每种温度下都会画一组柱子
x + i * bar_width 是为了错开三组柱子，不重叠
scaled_probas[i] 是当前温度下的 softmax 结果
label 用于图例，说明温度值
"""
for i, T in enumerate(temperatures):
    rects = ax.bar(x + i * bar_width, scaled_probas[i], bar_width, label=f'Temperature = {T}')
ax.set_ylabel('Probability')    #  y 轴标签
ax.set_xticks(x)    # 设置 x 轴位置
ax.set_xticklabels(vocab.keys(), rotation=90)   # 显示 token 名称，并旋转防重叠
ax.legend() # 显示图例
plt.tight_layout()  # 自动调整布局防止遮挡
plt.show()  #  显示图像


