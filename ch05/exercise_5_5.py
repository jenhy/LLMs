import torch
import tiktoken
import os, sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)  
sys.path.append(parent_dir)

from ch04.ch04_04_06 import GPTModel
from ch05.ch05_01_03 import get_data_loaders, calc_loss_loader
from ch05.ch05_05_01 import model_configs, load_weights_into_gpt
from gpt_download import download_and_load_gpt2



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

if __name__ == "__main__":
    train_loader, val_loader, tokenizer = get_data_loaders()
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    tokenizer = tiktoken.get_encoding("gpt2")

    model_name = "gpt2-small (124M)"
    NEW_CONFIG = GPT_CONFIG_124M.copy()
    NEW_CONFIG.update(model_configs[model_name])
    NEW_CONFIG.update({"context_length":1024, "qkv_bias":True})

    gpt = GPTModel(NEW_CONFIG)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpt.eval()
    print(NEW_CONFIG)
    
    settings, params = download_and_load_gpt2("124M", models_dir="gpt2")
    load_weights_into_gpt(gpt, params)
    gpt.to(device)

    torch.manual_seed(123)
    train_losss = calc_loss_loader(train_loader, gpt, device)
    val_losss = calc_loss_loader(val_loader, gpt, device)

    # Training loss: 3.7547490331861706
    print("Training loss:", train_losss)

    # Validation loss: 3.5596203804016113
    print("Validation loss:", val_losss)

    """
    解释：
    “The Verdict”并未包含在 OpenAI 训练的 GPT-2的预训练数据集中。因此，模型并没有显著地过拟合训练集。为什么？
    OpenAI 在训练 GPT-2 的时候，用的是一个大规模的语料库（比如 WebText）。
    如果你现在用一个GPT-2 没看过的新文本（比如“The Verdict”）作为数据集，那么：
    GPT-2 没有见过这些句子，所以不会过拟合。
    它在训练集和验证集上的表现就会非常接近。
    因为它对这两者都“陌生”，只是用已有的语言知识在做预测。

    为什么验证损失会略低于训练损失？
    虽然通常训练损失更低（因为模型直接优化的是它），
    但在小数据集上，有时验证集恰好比训练集更“容易预测”，就会出现这个结果，
    这种情况往往是偶然的波动（随机噪声），不是模型的真正能力差异。

    “The Verdict”是 GPT-2 的预训练数据之一。
    假如“The Verdict”确实在 GPT-2 的训练语料里，那情况就不同了。
    那么 GPT-2 在预测它时就是在“复述”它已经学过的内容。
    不仅训练集，它对验证集也可能“早就见过”，所以验证损失不会更高。

    为什么这有问题？
    你希望评估的是：GPT-2 在新数据上的泛化能力。
    如果验证集是它训练时见过的，那就不能算“泛化能力”的评估。
    所以你无法判断它是否过拟合了训练集，因为验证集也帮了它。
    """
 