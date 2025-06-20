import torch
import tiktoken
import os, sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)  
sys.path.append(parent_dir)

from ch04.ch04_04_06 import GPTModel
from ch05.ch05_01_01 import text_to_token_ids, token_ids_to_text
from ch05.ch05_01_03 import get_data_loaders
from ch05.ch05_02_01 import train_model_simple

def generate(model, idx, max_new_tokens, context_size, temperature=0.0, top_k=None, eos_id=None):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]
        if top_k is not None:
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]
            logits = torch.where(logits < min_val, torch.tensor(float('-inf')).to(logits.device), logits)
        if temperature > 0.0:
            logits = logits / temperature
            probs = torch.softmax(logits, dim=-1)
            torch.multinomial(probs, num_samples=1)
            idx_next = torch.multinomial(probs, num_samples=1)
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)
        if idx_next == eos_id:
            break
        idx = torch.cat((idx, idx_next), dim=1)
    return idx

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
    torch.manual_seed(123)
    model = GPTModel(GPT_CONFIG_124M)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   # 选择运算设备。电脑支持 CUDA（即有可用的 NVIDIA GPU），就用 GPU；否则退回 CPU。
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0004, weight_decay=0.01)
    train_losses, val_losses, tokens_seen = train_model_simple(model, train_loader, val_loader, optimizer, device, num_epochs=10, eval_freq=5, eval_iter=5, start_context="Every effort moves you", tokenizer=tokenizer)

    token_ids = generate(model=model, idx=text_to_token_ids("Every effort moves you", tokenizer), max_new_tokens=15, context_size=GPT_CONFIG_124M["context_length"], top_k=25, temperature=1.4)

    print("Output text:\n", token_ids_to_text(token_ids, tokenizer))
