
import torch
import tiktoken
import os, sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)  
sys.path.append(parent_dir)

from ch04.ch04_04_06 import GPTModel

def generate_text_simple(model, idx, max_new_tokens, context_size):
    r"""
    使用PyTorch为语言模型简单生成文本的函数。这个函数希望在已有的 idx 上继续生成新的 token，控制上下文长度是为了兼容模型的输入窗口限制。
    参数：
    model: 一个语言模型，输入 token 序列，输出下一个 token 的概率分布。
    idx: 当前的 token 序列（形状通常为 [batch_size, seq_len]）。
    max_new_tokens: 生成文本的最大长度。
    context_size: 每次送入模型的最大上下文长度（截断输入）。例如：模型只支持5个词元，此时文本长度为10，那么只有最后5个词元会被送入模型。
    返回：
    生成的文本的索引，一个二维的PyTorch张量。

    这个函数实现了一个简单的文本生成器，基于以下流程：
    1.每次取最后 context_size 个 token 作为输入；
    2.使用模型预测下一个 token 的分布；
    3.用 argmax 选出概率最大的 token；
    4.把这个 token 拼接到序列末尾；
    5.重复步骤直到生成足够的 token。
    这是典型的 贪婪解码 策略，不涉及随机性或多样性采样（如 Top-k、Top-p），适合快速、确定性的文本生成。
    """

    # 循环生成新 token，每次1个，重复 max_new_tokens次。模型一次只能预测下一个 token，因此需要循环逐个生成，直到达到想要的数量。
    # 这里的_表示的是一个变量，跟i，j或其他变量名一样，但是我并不关心当前是第几次，只想重复执行 max_new_tokens 次操作，所以用了_表示，是Python的惯用用法。
    for _ in range(max_new_tokens):

        # 从当前 idx 中取出最近的context_size个token。使用张量切片操作 [:, -context_size:] 保留最后的上下文窗口。模型只需关注最近的一段上下文（截断）
        # ：表示所有行，第一个维度是batch_size，表示“选取所有样本”。-context_size: 表示从倒数第 context_size 个到最后。例如，如果 context_size = 4，那么这相当于取最后的 4 个 token。
        idx_cond = idx[:, -context_size:]

        # 禁用梯度计算的上下文管理器。torch.no_grad() 关闭自动求导机制，在该块内不构建计算图。推理时不需要计算梯度，可以节省内存和加快推理速度。
        with torch.no_grad():

            # 用模型计算给定上下文的输出 logits。模型接收 token 序列 idx_cond 并输出预测 logits（形状通常为 [batch_size, seq_len, vocab_size]）。
            # logits 是未归一化的预测得分，后续将通过 softmax 得到概率分布，用于选择下一个 token。
            logits = model(idx_cond)

        # 取出模型输出中最后一个位置的logits。因此形状会从 [batch_size, seq_len, vocab_size] 变为 [batch_size, vocab_size]。
        # 使用张量索引操作：[:, -1, :] 表示每个样本的最后一个 token 的 logits（即预测下一个 token 的分布）。
        # 第一个维度：所有样本（batch）；第二个维度：序列的最后一个 token，-1 表示倒数第一个位置，也就是上下文中最后一个 token 的预测结果；第三个维度：词表维，保留所有 token 的 logits
        # 我们只关心下一个 token 的概率分布，而不是整个序列中每个 token 的分布。
        logits = logits[:, -1, :]

        # 将 logits 转换为概率分布。使用 softmax 函数按最后一个维度（词汇维vocab_size）进行概率归一化。gpt-2模型初始概率为1/vocab_size，也就是1/50257
        # softmax 能将 logits 转化为符合概率分布的值（和为1），这样我们可以用它来选择下一个 token。
        # probas 的形状为 [batch_size, vocab_size]，表示每个样本的概率分布。
        probas = torch.softmax(logits, dim=-1)
        # print("Probas shape:", probas.shape)

        # 选择概率最高的 token 作为下一个 token。argmax 找出最大概率的索引，keepdim=True 保留维度以方便拼接。
        # 这是贪婪解码（greedy decoding）的实现方式：每次选择最有可能的 token。
        # idx_next 的形状为 [batch_size, 1]，表示每个样本的下一个 token。
        idx_next = torch.argmax(probas, dim=-1, keepdim=True)
        # print("idx_next shape:", idx_next.shape)

        # 将新生成的 token 拼接到原序列末尾。使用 torch.cat 沿序列维（dim=1）拼接。生成的新 token 要加到序列里，作为下次预测的上下文。
        idx = torch.cat((idx, idx_next), dim=1)

    # 返回生成后的完整 token 序列。函数结束时将包含原始和新生成 token 的张量 idx 返回。
    return idx

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

if __name__ == "__main__":
    start_context = "Hello, I am"
    # 调用 tiktoken 库来加载 GPT-2 模型的分词器（tokenizer）,模型不能直接处理文本字符串，只能处理数字（token ID）；
    tokenizer = tiktoken.get_encoding("gpt2")
    encoded = tokenizer.encode(start_context)
    print("encoded:", encoded)

    # 把 encoded 列表转换成 PyTorch 张量；
    # .unsqueeze(0) 的作用是在最前面添加一维，变成形状 [1, seq_len]，表示一个 batch。
    # 模型要求输入形状为 [batch_size, seq_len]，即使只处理一个句子，也要包装成 batch。
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    print("encoded_tensor:", encoded_tensor.shape)

    model = GPTModel(GPT_CONFIG_124M)

    # 把模型设置为“评估模式（eval mode）”。
    # 在推理（生成）时应关闭 dropout、batch norm 等训练相关机制；
    model.eval()
    out = generate_text_simple(model=model, idx=encoded_tensor, max_new_tokens=6, context_size=GPT_CONFIG_124M["context_length"])
    print("Output:", out)
    print("Output length:", len(out[0]))

    # 将模型生成的 token ID 序列还原为字符串；
    # .squeeze() 去掉 batch 维度；
    # .tolist() 转为 Python 列表；
    # tokenizer.decode(...) 转为英文句子。
    decoded_text = tokenizer.decode(out.squeeze().tolist())
    print("decoded_text:", decoded_text)
