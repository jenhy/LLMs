import torch
import gc
import sys, os
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# 添加上级路径以便导入自定义模块
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)  
sys.path.append(parent_dir)

from ch05.ch05_01_03 import calc_loss_batch, calc_loss_loader, get_data_loaders
from ch05.ch05_01_01 import text_to_token_ids, token_ids_to_text
from ch04.ch04_04_07 import generate_text_simple
from ch04.ch04_04_06 import GPTModel

def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    """
    评估模型在训练集和验证集上的表现。它会在每次模型更新后打印训练集和验证集的平均交叉熵损失，以便评估训练是否改善，判断是否过拟合或欠拟合。
    参数解释：
    model：GPT模型对象。
    train_loader：训练集加载器。
    val_loader：验证集加载器。
    device：设备（如 'cpu' 或 'cuda'），控制数据/模型的计算在哪进行。
    eval_iter：模型评估的迭代次数。
    返回：
    返回训练集和验证集的平均交叉熵损失。
    """

    model.eval()    # 评估模式。PyTorch 的模型对象有 train() 和 eval() 方法控制行为。在评估阶段禁用 dropout 和 batchnorm 等，使产生结果稳定。
    with torch.no_grad():   # 创建一个上下文管理器，不需要计算梯度。减少内存消耗，提高推理速度；评估时不需要梯度。
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)   # 用于计算训练集加载器中所有批次的平均交叉熵损失。
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)   # 用于计算验证集加载器中所有批次的平均交叉熵损失。
    model.train()   # 恢复模型为训练模式。
    return train_loss, val_loss #  返回训练集和验证集的平均交叉熵损失。

def generate_and_print_sample(model, tokenizer, device, start_context):
    """
    用于跟踪模型在训练过程中是否有所改进的函数。
    参数解释：
    model: 模型对象。
    tokenizer: 分词器对象。
    device: 设备（如 'cpu' 或 'cuda'），控制数据/模型的计算在哪进行。
    start_context: 模型开始的上下文。
    """
    model.eval()    # 模型设置为评估模式

    """获取模型的位置嵌入矩阵的第 0 维长度（即最大 token 长度）。
    也就是这行代码self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])，结果就是context_length。
    在生成文本时，需要控制生成的 token 数量不能超过模型支持的最大上下文长度。如果生成时输入长度 + 新生成 token 超过了这个限制，会导致位置超界，从而模型报错。"""
    context_size = model.pos_emb.weight.shape[0]
    encoded = text_to_token_ids(start_context, tokenizer).to(device)    # 将起始文本转换为 token id，并移动到模型设备。
    with torch.no_grad():   # 创建一个上下文管理器，不需要计算梯度。减少内存消耗，提高推理速度；评估时不需要梯度。
        token_ids = generate_text_simple(model=model, idx=encoded, max_new_tokens=50, context_size=context_size)    # 生成一个最多50个token的句子。
    decoded_text = token_ids_to_text(token_ids, tokenizer)  # 解码token为文本
    print(decoded_text.replace("\n", " "))  #  打印文本
    model.train()   # 恢复模型为训练模式。
def train_model_simple(model, train_loader, val_loader, optimizer, device, num_epochs, eval_freq, eval_iter, start_context, tokenizer):
    """
    预训练大模型的主函数。
    参数解释：
    model: 这里是GPT-2模型。
    train_loader: 训练数据加载器。
    val_loader: 验证数据加载器。
    optimizer: 优化器。
    device: 设备（如 'cpu' 或 'cuda'），控制数据/模型的计算在哪进行。
    num_epochs: 训练的轮数。
    eval_freq: 模型评估的频率。也就是多少步打印一下
    eval_iter:  模型评估的迭代次数。
    start_context: 模型开始的上下文。
    tokenizer: 分词器。
    返回：
    跟踪训练数据损失、验证数据损失和所见的词元。后续用于画图或日志，帮助可视化学习进度。
    """
    train_losses, val_losses, track_tokens_seen = [], [], []    # 初始化列表以跟踪损失和所见的词元。后续用于画图或日志，帮助可视化学习进度。
    tokens_seen, global_step = 0, -1    # 训练的token数和全局步数。如果把global_step=0，那第一次 global_step += 1 后会变成 1，那么训练日志会从 Step 1 开始。很多人更喜欢从 Step 0 开始（比如 TensorBoard、PyTorch Lightning、Transformer 库等都从 Step 0 开始）

    # 开始训练轮数
    for epoch in range(num_epochs):
        model.train()   # 训练模式
        for input_batch, target_batch in train_loader:  # 从训练数据中逐批加载输入和目标。train_loader 是一个迭代器，提供数据对。
            optimizer.zero_grad()   # 重置上一个批次迭代中的损失梯度。梯度默认会累积，所以每轮要清除旧的，然后更新模型。
            loss = calc_loss_batch(input_batch, target_batch, model, device)    # 用于计算通过训练集加载器中每个批次的交叉熵损失。
            loss.backward() # 通过反向传播计算损失梯度
            optimizer.step()    # 使用损失梯度更新模型权重
            tokens_seen += input_batch.numel()  # 累加当前batch中的token数
            global_step += 1    # 累积全局步数

            # 每隔 eval_freq 步评估一次。global_step先加再判断（-1 + 1 -> 0，第一步就评估），为了让 step=0 的时候就可以立刻评估或打印日志，更早地获得一次训练状态的反馈。为了不需要每batch都评估，代价大，所以定期即可。
            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(model, train_loader, val_loader, device, eval_iter) # 执行评估，并打印当前结果，保存记录。
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(f"Ep {epoch+1} (Step {global_step:06d}) Train loss {train_loss:.3f} Validation loss {val_loss:.3f}")

            del loss, input_batch, target_batch
            torch.cuda.empty_cache()
            gc.collect()
        generate_and_print_sample(model, tokenizer, device, start_context)  # 每轮结束后生成示例文本。
    return train_losses, val_losses, track_tokens_seen

def plot_losses(epochs_seen, tokens_seen, train_losses, val_losses):
    fig, ax1 = plt.subplots(figsize=(5, 3))
    ax1.plot(epochs_seen, train_losses, label="Training loss")
    ax1.plot(epochs_seen, val_losses, linestyle="-.", label="Validation loss")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right")
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax2 = ax1.twiny()
    ax2.plot(tokens_seen, train_losses, alpha=0)
    ax2.set_xlabel("Tokens seen")
    fig.tight_layout()
    plt.show()

# 定义字典，用于配置GPT-2模型参数
GPT_CONFIG_124M = {
    "vocab_size": 50257,    # 词汇表大小，被BPE分词器使用的由50257个单词组成的词汇表
    "context_length": 256,  # 上下文长度，能够处理的最大输入词元数量
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
    r"""
    设置优化器。使用 AdamW，设置学习率和权重衰减。AdamW 是训练 Transformer 的推荐方法。
    这是 PyTorch 中的优化器，一种改进版的 Adam 优化器，叫做 AdamW（Adam with decoupled Weight Decay）。
    它的特点是将 weight_decay（权重衰减）从梯度更新中分离出来，从而改进原始 Adam 中正则化与梯度冲突的问题。这在 训练 Transformer/GPT 等模型时非常重要。
    model.parameters()方法返回模型中所有需要优化的参数（即可训练参数）。
    lr=0.0004。学习率（Learning Rate），控制每次参数更新的步长。值越大：模型学习越快，但容易震荡甚至发散。值越小：训练更稳定，但收敛更慢。
    weight_decay=0.01。权重衰减系数，用于正则化模型，防止过拟合。它会在每次参数更新时惩罚权重太大的参数。

    为什么用 AdamW 而不是 Adam？
    在训练 Transformer、GPT、BERT 这类模型时，几乎所有论文和官方实现都推荐使用 AdamW，因为：
    Adam 把权重衰减混入梯度中，导致优化方向偏移；
    AdamW 分离了梯度和正则项，更新更稳定，更容易收敛。
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0004, weight_decay=0.01)
    train_losses, val_losses, tokens_seen = train_model_simple(model, train_loader, val_loader, optimizer, device, num_epochs=10, eval_freq=5, eval_iter=5, start_context="Every effort moves you", tokenizer=tokenizer)

    num_epochs = 10
    epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
    plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)

    """
    epochs 10轮训练结果：

    Characters: 20479
    Tokens: 5145
    Train batches: 9, Val batches: 1
    Ep 1 (Step 000000) Train loss 9.783 Validation loss 9.927
    Ep 1 (Step 000005) Train loss 7.984 Validation loss 8.334
    Every effort moves you,,,,,,,,,,,,.
    Ep 2 (Step 000010) Train loss 6.753 Validation loss 7.047
    Ep 2 (Step 000015) Train loss 6.113 Validation loss 6.573
    Every effort moves you, and,, and, and,,,,, and, and,,,,,,,,,,,,,, and,,,, and,, and,,,,, and,,,,,,
    Ep 3 (Step 000020) Train loss 5.523 Validation loss 6.491
    Ep 3 (Step 000025) Train loss 5.322 Validation loss 6.386
    Every effort moves you, and to the picture.                      "I, and the of the of the's the honour, and, and I had been, and I
    Ep 4 (Step 000030) Train loss 4.755 Validation loss 6.361
    Ep 4 (Step 000035) Train loss 4.458 Validation loss 6.262
    Every effort moves you of the to the picture to the of the picture--and by his of the picture to have to        "I was his the picture and I had the picture.
    Ep 5 (Step 000040) Train loss 3.825 Validation loss 6.198
    Every effort moves you know the "Oh, and he was not the fact by his last word.         "I was.      "Oh, I felt a little a little the    
    Ep 6 (Step 000045) Train loss 3.341 Validation loss 6.140
    Ep 6 (Step 000050) Train loss 2.848 Validation loss 6.112
    Every effort moves you know; and my dear, and he was not the fact with a little of the house of the fact of the fact, and.
    Ep 7 (Step 000055) Train loss 2.331 Validation loss 6.138
    Ep 7 (Step 000060) Train loss 2.066 Validation loss 6.181
    Every effort moves you know," was one of the picture for nothing--I told Mrs.  "I looked--as of the fact, and I felt him--his back his head to the donkey. "Oh, and_--because he had always _
    Ep 8 (Step 000065) Train loss 1.503 Validation loss 6.180
    Ep 8 (Step 000070) Train loss 1.252 Validation loss 6.182
    Every effort moves you?" "I didn't bear the picture--I told me.  "I looked up, and went on groping and Mrs. I was back the head to look up at the honour being _mine_--because he didn't want
    Ep 9 (Step 000075) Train loss 0.980 Validation loss 6.282
    Ep 9 (Step 000080) Train loss 0.702 Validation loss 6.288
    Every effort moves you?"  "Yes--quite insensible to the irony. She wanted him vindicated--and by me!"  He laughed again, and threw back his head to look up at the sketch of the donkey. "There were days when I
    Ep 10 (Step 000085) Train loss 0.491 Validation loss 6.335
    Every effort moves you?"  "Yes--quite insensible to the irony. She wanted him vindicated--and by me!"  He laughed again, and threw back his head to the donkey again. I saw that, when Stroud laid in the first
    
    """