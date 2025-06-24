import tiktoken # 导入OpenAI的BPE分词器，用于将文本转为token序列
import torch
import os, sys

# 添加上级路径以便导入自定义模块
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)  
sys.path.append(parent_dir)

# 导入自定义模块
from ch02.ch02_06 import create_dataloader_v1   # 用于创建PyTorch的数据加载器（DataLoader），适配GPT模型训练。
from ch04.ch04_04_06 import GPTModel    # 自定义GPT架构模型，仿照GPT-2构建。


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


def calc_loss_batch(input_batch, target_batch, model, device):
    r"""
    用于计算通过训练集加载器和验证集加载器中每个批次的交叉熵损失。
    参数解释：
    input_batch：输入token的张量（形状如 [B, T]，B是batch size，T是token序列长度context_length）。
    target_batch：目标token的张量（通常是input向右移动一位，即语言建模任务的“下一个词预测”）。形状同input_batch。
    model：GPT模型对象。
    device：设备（如 'cpu' 或 'cuda'），控制数据/模型的计算在哪进行。
    """

    # to(device)：将数据移动到指定的设备上，如GPU或CPU。PyTorch中的模型和数据必须在同一设备上才能计算（例如，不能把数据放在CPU而模型在GPU，否则会报错）。
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)

    # 前向传播：计算预测结果（logits）。输入形状为[B, T]，其中B是batch size，T是token序列长度context_length。输出形状为[B, T, vocab_size]，其中B是batch size，T是token序列长度context_length，vocab_size是词汇表大小。
    # 如果是语言模型，预测所有词元的预测结果，因此取model(input_batch)
    # 如果是分类任务，只关注最后一个词元的预测结果，因此取model(input_batch)[:, -1, :]，输出形状为[B, num_classes]，其中B是batch size，num_classes是类别数量。
    logits = model(input_batch).to(device)
    # logits = model(input_batch)[:, -1, :]
    # print("logits.shape:", logits.shape)
    # print("target.shape:", target_batch.shape)

    print(f"[DEBUG] input_batch.device: {input_batch.device}, target_batch.device: {target_batch.device}, logits.device: {logits.device}")

    # cross_entropy：计算交叉熵损失。PyTorch 的 F.cross_entropy() 会自动对 logits 进行 softmax，然后计算预测值与真实标签之间的负对数似然损失（NLLLoss）。
    # 参数解释：
    # logits.flatten(0, 1)：将logits张量展平，形状为[B*T, vocab_size]，其中B是batch size，T是token序列长度context_length，vocab_size是词汇表大小。
    # target_batch.flatten()：将target_batch张量展平，形状为[B*T]，其中B是batch size，T是token序列长度context_length。
    # 如果是语言模型，需要展平，即logits.flatten(0, 1), target_batch.flatten()
    # 如果是分类任务，不需要展平，即logits, target_batch，形状分别为[B, num_classes]、[B]

    loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
    # loss = torch.nn.functional.cross_entropy(logits, target_batch)

    # 返回这个 batch 的平均损失
    return loss

def calc_loss_loader(data_loader, model, device, num_batches=None):
    r"""
    用于计算训练集加载器和验证集加载器中所有批次的平均交叉熵损失。
    参数解释：
    data_loader：训练集加载器train_loader或验证集加载器val_loader。
    model：GPT模型对象。
    device：设备（如 'cpu' 或 'cuda'），控制数据/模型的计算在哪进行。
    num_batches：可选参数，用于限制计算损失时使用的批次数量。如果为 None，则使用所有批次。
    返回值：
    平均交叉熵损失。
    """

    # 初始化总损失
    total_loss = 0.

    # 如果没有数据，直接返回 NaN。因为没有数据就没有办法计算损失，这种情况是非法的或代表数据错误，NaN 是一种提醒。
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)  # 如果没有指定批次数量，则使用所有批次
    else:
        num_batches = min(num_batches, len(data_loader))    # 如果指定了批次数量，则使用最小值，为了防止超出实际的 batch 数量

    # 遍历每个 batch 来计算损失
    # i 是当前的 batch 编号。(input_batch, target_batch) 是一对输入张量和目标标签张量。
    # 如果当前 batch 编号还在要处理的范围之内，调用前面定义的 calc_loss_batch(...) 函数来计算当前 batch 的损失。
    # .item() 把 PyTorch 的张量转成 Python 数字，方便加总。如：tensor(53852.2812)转换为53852.2812
    # 累加到 total_loss 上。
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break

    #  返回平均损失
    return total_loss / num_batches

def get_data_loaders():
    """
    获取数据加载器的主函数，可以被其他模块导入使用
    """
    # 读取文本数据，保存到text_data字符串变量中
    file_path = f"C:\\Users\\Jenhy\\OneDrive\\doc\\学习\\AI\\LLMs\\ch02\\the-verdict.txt"
    with open(file_path, "r", encoding="utf-8") as file:
        text_data = file.read()

    # 统计原始的字符数和Token数量
    # 使用 GPT-2 的 BPE 分词器将文本分词并计算 token 数（注意：token 数通常 < 字符数，因为BPE合并了一些常见词块）。
    total_characters = len(text_data)
    print(f"Characters: {total_characters}")

    tokenizer = tiktoken.get_encoding("gpt2")
    total_tokens = len(tokenizer.encode(text_data))
    print(f"Tokens: {total_tokens}")

    # 将原始文本数据划分为训练集和验证集
    train_ratio = 0.90
    split_idx = int(train_ratio * total_characters)
    train_data = text_data[:split_idx]
    val_data = text_data[split_idx:]

    # 设置随机种子，确保每次运行时，结果都是相同的
    torch.manual_seed(123)

    # 创建数据加载器DataLoader
    r"""
    train_data: 训练集，包含所有原始文本数据
    batch_size: 训练时每次输入的样本数量
    max_length: 输入的文本最大长度，即每个样本的token数量
    stride: 滑动窗口的步长。每条数据之间有重叠，用了滑动窗口。当stride=max_length时，相当于没有重叠。然后把这些样本按batch_size为一组打包成 DataLoader。
    drop_last: 是否丢弃最后一个不完整的batch，默认为True，表示丢弃最后一个不完整的batch
    shuffle: 是否打乱训练集的顺序，默认为True，表示打乱训练集的顺序
    """
    train_loader = create_dataloader_v1(train_data, batch_size=2, max_length=GPT_CONFIG_124M["context_length"], stride=GPT_CONFIG_124M["context_length"], drop_last=True, shuffle=True, num_workers=0)

    val_loader = create_dataloader_v1(val_data, batch_size=2, max_length=GPT_CONFIG_124M["context_length"], stride=GPT_CONFIG_124M["context_length"], drop_last=False, shuffle=False, num_workers=0)

    return train_loader, val_loader, tokenizer

if __name__ == "__main__":
    train_loader, val_loader, _ = get_data_loaders()

    """
    x.shape = [2, 256]：表示这是一个 batch，包含 2 个样本，每个样本是 256 个 token 的输入
    y.shape = [2, 256]：是对应的目标输出（通常是 x 向右 shift 一个位置）,这个实现在create_dataloader_v1里的GPTDatasetV1函数里

    Train loader:
    torch.Size([2, 256]) torch.Size([2, 256])
    torch.Size([2, 256]) torch.Size([2, 256])
    torch.Size([2, 256]) torch.Size([2, 256])
    torch.Size([2, 256]) torch.Size([2, 256])
    torch.Size([2, 256]) torch.Size([2, 256])
    torch.Size([2, 256]) torch.Size([2, 256])
    torch.Size([2, 256]) torch.Size([2, 256])
    torch.Size([2, 256]) torch.Size([2, 256])
    torch.Size([2, 256]) torch.Size([2, 256])

    Validation loader:
    torch.Size([2, 256]) torch.Size([2, 256])

    """
    print(f"Train loader:")
    for x, y in train_loader:
        print(x.shape, y.shape)

    print(f"\nValidation loader:")
    for x, y in val_loader:
        print(x.shape, y.shape)

    model = GPTModel(GPT_CONFIG_124M)   # 实例化模型，并移动到设备上

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   # 选择运算设备。电脑支持 CUDA（即有可用的 NVIDIA GPU），就用 GPU；否则退回 CPU。
    model.to(device)    # 把模型放到运算设备上（CPU 或 GPU）。

    # 关闭梯度计算，提高速度,节省内存。在验证或评估阶段，不需要做反向传播（backpropagation）
    """
    这里用了with的上下文管理器Context Manager，是 Python 中用于 管理资源的一种机制。
    可以方便、安全地分配和释放资源，比如：打开/关闭文件；连接/断开数据库；加锁/解锁线程；开启/关闭梯度计算（像 torch.no_grad()）；临时修改状态（比如 numpy 精度设置）等。

    语法：
    with some_context_manager:
        # 这里是上下文环境内部，资源已分配
        ...
    # 出了这个代码块，资源会被自动释放

    举例：
    with open("file.txt", "r") as f:
        data = f.read()
    等价于：
    f = open("file.txt", "r")
    try:
        data = f.read()
    finally:
        f.close()

    with 会自动调用：
    __enter__() 方法 → 进入资源状态；
    __exit__() 方法 → 退出时自动清理资源（比如关闭文件）。

    自定义上下文管理器：
    class MyContext:
        def __enter__(self):
            print("进入上下文")
            return self
        def __exit__(self, exc_type, exc_value, traceback):
            print("退出上下文")

    with MyContext():
        print("处理中…")

    """
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device)
        val_loss = calc_loss_loader(val_loader, model, device)
    # Training loss: 10.99745644463433
    # Validation loss: 11.03940486907959
    print("Training loss:", train_loss)
    print("Validation loss:", val_loss)
