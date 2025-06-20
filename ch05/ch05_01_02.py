import torch
import tiktoken
import os, sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)  
sys.path.append(parent_dir)

r"""current_dir: c:\Users\Jenhy\OneDrive\doc\学习\AI\LLMs\ch05, parent_dir: c:\Users\Jenhy\OneDrive\doc\学习\AI\LLMs, sys.path: ['c:\\Users\\Jenhy\\OneDrive\\doc\\学习\\AI\\LLMs\\ch05', 'D:\\Program Files\\Python\\Python38\\python38.zip', 'D:\\Program Files\\Python\\Python38\\DLLs', 'D:\\Program Files\\Python\\Python38\\lib', 'D:\\Program Files\\Python\\Python38', 'D:\\Program Files\\Python\\Python38\\lib\\site-packages', 'c:\\Users\\Jenhy\\OneDrive\\doc\\学习\\AI\\LLMs']"""
# print(f"current_dir: {current_dir}, parent_dir: {parent_dir}, sys.path: {sys.path}")

from ch04.ch04_04_06 import GPTModel
from ch04.ch04_04_07 import generate_text_simple
from ch05.ch05_01_01 import token_ids_to_text, text_to_token_ids

# 定义字典，用于配置GPT-2模型参数
GPT_CONFIG_124M = {
    "vocab_size": 50257,    # 词汇表大小，被BPE分词器使用的由50257个单词组成的词汇表
    "context_length": 256,  # 上下文长度，能够处理的最大输入词元数量，将1024个词元改成256个，减少训练模型的计算需求
    "emb_dim": 768,         # 嵌入维度，每个词元转化为768维向量
    "n_heads": 12,          # 多头注意力机制中注意力头的数量
    "n_layers": 12,         # Transformer块的数量
    "drop_rate": 0.1,       # 丢弃率10%，在训练过程中，随机丢弃一些神经元，以减少过拟合
    "qkv_bias": False       # 是否使用偏置项。在多头注意力机制的线性层中添加一个偏置项，用于查询、键和值的计算
}

torch.manual_seed(123)
model = GPTModel(GPT_CONFIG_124M)

## 第一步：将输入映射为词元ID
"""
将以下两个输入示例映射为词元ID。
["every effort moves", "I really like"]
"""
inputs = torch.tensor([[16833, 3626, 6100],
               [40, 1107, 588]])

## 第二步：预期生成结果
"""
我希望模型给我生成的词元ID。
["effort moves you", "really like chocolate"]
"""
targets = torch.tensor([[3626, 6100, 345],
               [1107, 588, 11311]])

## 第三步：生成未归一化的分数logits
"""
因为我们还没有开始训练模型，所以现在我们只使用无梯度的上下文。也就是屏蔽模型参数的梯度跟踪。
"""
with torch.no_grad():
    logits = model(inputs)

# logits.shape:torch.Size([2, 3, 50257])
print(f"logits.shape:{logits.shape}") 

r"""
tensor([[[ 0.0970, -0.4086, -0.5987,  ...,  0.4720, -0.5374,  0.2756],
         [-0.5862, -0.6020, -0.8016,  ...,  0.3963, -1.0509,  0.2093],
         [ 0.5252, -0.7590, -0.3648,  ...,  0.2666, -0.7289,  0.2032]],

        [[ 0.0823,  0.3897, -0.7336,  ..., -0.2326,  0.3569, -0.1017],
         [-0.7512, -0.4674, -0.7684,  ...,  0.1094, -0.3054, -0.4782],
         [ 0.4637,  0.5048,  0.6473,  ..., -0.7839,  0.8850, -0.1714]]])
"""
# print(logits)

# 词汇表中每个词元的概率
probas = torch.softmax(logits, dim=-1)
# torch.set_printoptions(sci_mode=False)
r"""
tensor([[[1.8608e-05, 1.1224e-05, 9.2801e-06,  ..., 2.7075e-05,
          9.8672e-06, 2.2246e-05],
         [9.4048e-06, 9.2574e-06, 7.5821e-06,  ..., 2.5121e-05,
          5.9095e-06, 2.0838e-05],
         [2.8551e-05, 7.9050e-06, 1.1725e-05,  ..., 2.2044e-05,
          8.1464e-06, 2.0690e-05]],

        [[1.8320e-05, 2.4914e-05, 8.1020e-06,  ..., 1.3371e-05,
          2.4109e-05, 1.5242e-05],
         [7.9197e-06, 1.0518e-05, 7.7847e-06,  ..., 1.8726e-05,
          1.2369e-05, 1.0405e-05],
         [2.6762e-05, 2.7883e-05, 3.2154e-05,  ..., 7.6854e-06,
          4.0780e-05, 1.4179e-05]]])
"""
# print(probas)

# [batch_size, seq_len, vocab_size]
# torch.Size([2, 3, 50257])
print(probas.shape)

## 第四步：将概率映射为词元ID
token_ids = torch.argmax(probas, dim=-1, keepdim=True)

"""
Token IDs:
 tensor([[[ 6065],
         [13866],
         [44376]],

        [[49906],
         [ 3684],
         [38185]]])
"""
print("Token IDs:\n", token_ids[0])

# torch.Size([2, 3, 1])
print("Token IDs shape:\n", token_ids.shape)

tokenizer = tiktoken.get_encoding("gpt2")

## 第五步：将词元ID映射为文本
"""从输出词元与我们希望的目标输出词元进行比较，发现完全不同，因为我们还没有进行训练。
Targets batch 1: effort moves you
Outputs batch 1: Menbage inaction
其中flatten()函数将多维张量降维为一维张量。
token_ids[0]的形状为[3, 1]，使用flatten()函数会将 [3, 1] 变成 [3]，
也就是将 tensor([[ 6065],
        [13866],
        [44376]])
拉成一维数组tensor([6065, 13866, 44376])
其中flatten()函数用来合并多个维度。
这个名字 flatten 的由来，其实就是字面意义 —— “压平”、“展平” 的意思。从视觉或结构上理解，一个多维的张量（tensor）就像是一个“嵌套的”结构，比如列表套列表，或者矩阵堆叠矩阵。而 flatten 做的事情，就是把这些嵌套的结构 “压平”，变成一个线性的或更简单的结构。
"""
print(f"Targets batch 1:{token_ids_to_text(targets[0], tokenizer)}")
print(f"Outputs batch 1:{token_ids_to_text(token_ids[0].flatten(), tokenizer)}")

## 第六步：打印目标概率分布
"""
取模型预测结果中对应目标词targets的概率分布probas。其中targets是我们提前准备的“正确答案”词元ID张量，告诉模型你应该生成什么。probas是模型根据inputs输出的预测概率分布，给出每个位置对所有词的预测概率。

probas的形状为 [2, 3, 50257]，表示 batch_size=2，seq_len=3，vocab_size=50257
以第一个文本为例，text_idx=0：
text_idx 是一个标量。这里是第0个文本（文本维度）。
[0, 1, 2] 是一个列表。这里是第0个文本的第0个词、第1个词、第2个词（位置维度）。
targets[text_idx] 是一个列表或张量。这里是第0个文本的词元ID（词汇表维度）。

举例：
probas[text_idx, [0, 1, 2], targets[text_idx]] 是PyTorch 的[行, 列, 值]高级索引/广播索引（Advanced Indexing），允许我们同时沿多个维度进行复杂的选择。
等价于：
[
  probas[0, 0, targets[0][0]],
  probas[0, 1, targets[0][1]],
  probas[0, 2, targets[0][2]]
]
也就是：
模型对第1个词位置上“真实词3626”的预测概率，即probas[0, 0, 3626]；
模型对第2个词位置上“真实词6100”的预测概率，即probas[0, 1, 6100]；
模型对第3个词位置上“真实词345”的预测概率，即probas[0, 2, 345]。
所以target_probas_1 = [probas[0, 0, 3626], probas[0, 1, 6100], probas[0, 2, 345]] = tensor([4.3559e-05, 2.2607e-05, 1.1023e-05])

说白了就是拿着“目标词元的索引”，去已经计算好的probas张量里查一下它们的概率值是多少。
"""
text_idx = 0
target_probas_1 = probas[text_idx, [0, 1, 2], targets[text_idx]]
print("Text 1:", target_probas_1)   # Text 1: tensor([4.3559e-05, 2.2607e-05, 1.1023e-05])

text_idx = 1
target_probas_2 = probas[text_idx, [0, 1, 2], targets[text_idx]]
print("Text 2:", target_probas_2)   # Text 2: tensor([1.3012e-05, 6.0395e-05, 4.8063e-06])

## 第七步：计算概率分数的损失

# 对概率分数应用对数
# cat后的结果为：tensor([4.3559e-05, 2.2607e-05, 1.1023e-05, 1.3012e-05, 6.0395e-05, 4.8063e-06])
# log(cat(...))的结果为：tensor([-10.0414, -10.6972, -11.4156, -11.2496,  -9.7146, -12.2456])
log_probas = torch.log(torch.cat((target_probas_1, target_probas_2)))
print(log_probas)

# 平均对数概率分数
# tensor(-10.8940)
avg_log_probas = torch.mean(log_probas)
print(avg_log_probas)

# 负平均对数概率分数=平均对数概率分数乘以-1，这个将负值转换为正值称为交叉熵损失。
# 在pytorch中，交叉熵损失函数torch.nn.fuunctional.cross_entropy()会自动计算交叉熵损失。
# tensor(10.8940)
neg_avg_log_probas = avg_log_probas * -1
print(neg_avg_log_probas)

## 第八步：使用flatten函数来展平张量
"""其中flatten()函数用来合并多个维度。
这个名字 flatten 的由来，其实就是字面意义 —— “压平”、“展平” 的意思。从视觉或结构上理解，一个多维的张量（tensor）就像是一个“嵌套的”结构，比如列表套列表，或者矩阵堆叠矩阵。而 flatten 做的事情，就是把这些嵌套的结构 “压平”，变成一个线性的或更简单的结构。
将 logits 的第 0 维和第 1 维 合并为一个维度，其余维度保持不变。
targets 是将所有维度展平成一维。
"""
logits_flat = logits.flatten(0, 1)
targets_flat = targets.flatten()
print("Flattened logits shape:", logits_flat.shape) # Flattened logits shape: torch.Size([6, 50257])
print("Flattened targets shape:", targets_flat.shape) # Flattened targets shape: torch.Size([6])


## 第九步：将第六步和第七步用PyTorch的cross_entropy()交叉熵损失函数替换
"""
cross_entropy()交叉熵损失函数是对模型输出 logits_flat 和真实标签 targets_flat 之间计算 交叉熵损失。
参数1：input 或叫 logits_flat，这是模型的输出，也叫 logits，形状为：(N, C)，N: 样本个数（例如：batch_size × seq_len），C: 类别数（比如：词表大小 vocab_size）。注意：这个张量不能是 softmax 输出，而应是模型的原始输出（未归一化的 logit）。cross_entropy 会内部自动对它进行 softmax 处理。
参数2：target 或叫 targets_flat，这是模型对应的真实标签，形状为：(N,)
"""
loss = torch.nn.functional.cross_entropy(logits_flat, targets_flat)
# tensor(10.8940)
print(loss)

## 第十步：计算困惑度
"""
将交叉熵损失（cross entropy loss）转换为 困惑度（perplexity）。
Perplexity（困惑度） 是衡量一个语言模型对一段文本预测能力的指标，越小表示模型越“自信”，越好。
这个结果可以理解为模型对每个词的 平均预测分支数，也就是：
如果 perplexity = 1，说明模型每次都预测得非常准（100% 置信度）；
如果 perplexity = 10，说明模型“在10个候选中猜一个”；
如果 perplexity = vocab_size，说明模型完全不会预测（相当于瞎猜）。
"""
perplexity = torch.exp(loss)
# tensor(53852.2812)
# 词汇表vocab_size大小是50257，现在的 perplexity是53852，说明：模型的预测非常差，几乎是“瞎猜”的状态，甚至比完全随机选择还要差。
print(perplexity)