from ch02_06 import create_dataloader_v1
import torch

with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()
print("原始文本长度:", len(raw_text))

dataloader = create_dataloader_v1(raw_text, batch_size=8, max_length=4, stride=4, shuffle=False)
data_iter = iter(dataloader)
inputs, targets = next(data_iter) # next(data_iter) 返回的是一个元组，根据GPTDatasetV1 的 __getitem__ 返回的
print("Token IDs:\n", inputs)
print("\nInputs shape:\n", inputs.shape)

# 创建一个词元嵌入层，将词元ID映射为向量
vocab_size = 50257 # GPT-2 的词表大小
output_dim = 256
token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
token_embeddings = token_embedding_layer(inputs)
print(token_embeddings.shape)   # 输出形状为：8 * 4 * 256，8 个样本；每个样本 4 个 token；每个 token 映射成一个 256 维向量。

# 创建一个位置嵌入层，将 0～3 的位置 index 映射为向量
context_length = 4
pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)
print(torch.arange(context_length)) # tensor([0, 1, 2, 3])
pos_embeddings = pos_embedding_layer(torch.arange(context_length))
print(pos_embeddings.shape) # 输出形状为：4 * 256，表示每个位置一个 256 维向量

input_embeddings = token_embeddings + pos_embeddings # 将词元嵌入向量和位置嵌入向量相加和得到最终输入嵌入向量，运用了广播机制，相当于8 * 4 * 256 + 1 * 4 * 256 = 8 * 4 * 256
print(input_embeddings.shape)   # 输出形状为：8 * 4 * 256

flat_embedding = input_embeddings.view(-1, input_embeddings.size(-1))   # 将输入嵌入向量展平，得到一个二维张量，每行是一个样本的输入嵌入向量。[8, 4, 256] 拉平成 [32, 256]
# torch.set_printoptions(threshold=float('inf')) # 设置打印选项为“无限阈值”，显示所有元素
print(flat_embedding)   # 输出前 5 行

# 总结流程：文本 → 分词 → Token ID → Token Embedding → + Position Embedding → 输入 Transformer