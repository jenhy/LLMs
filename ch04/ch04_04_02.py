import torch    # 是 PyTorch 的核心库，提供张量计算、自动求导、模型训练等功能。
import torch.nn as nn   # torch.nn 是构建神经网络的模块库，nn.Module, nn.Linear, nn.ReLU, nn.Sequential 等都在这里


class LayerNorm(nn.Module):
    r"""
    定义一个自定义层归一化模块，继承自 nn.Module，使其可以像 PyTorch 模块一样使用。封装成模块可以和其他层集成起来，支持 GPU 运算、参数管理、梯度反向传播等。
    """
    def __init__(self, emb_dim):
        r"""
        __init__ 是类的初始化函数，emb_dim 是输入张量的最后一个维度大小（例如 [batch_size, seq_len, emb_dim]）。
        super().__init__() 调用父类的初始化方法，是必须的，否则这个模块无法被正确注册到模型中，影响模型参数管理和保存。
        """
        super().__init__()
        self.eps = 1e-5 #  epsilon是一个小常数，在归一化的过程中被加到方差上，防止分母为零

        r"""
        归一化会将激活变为标准正态分布（均值0，方差1），但这样可能限制了网络表达能力，引入缩放参数和偏移参数，可以让模型在需要时恢复原始分布或学习更复杂的变化
        """
        self.scale = nn.Parameter(torch.ones(emb_dim))  #  scale 是可学习的缩放参数，初始化全为1，形状为 [emb_dim]，对每个维度单独调整
        self.shift = nn.Parameter(torch.zeros(emb_dim)) #  shift 是可学习的偏移参数，初始化为全0，形状为 [emb_dim]，对每个维度单独调整

    def forward(self, x):
        r"""
        定义前向传播（也就是当执行 layer(x) 时会调用这个函数）。
        这是 nn.Module 类最核心的方法，PyTorch 会在训练和推理时自动调用它。
        """
        mean = x.mean(dim=-1, keepdim=True) # 对最后一个维度（通常是 embedding 维度）求均值，层归一化是对每一个样本的每一层特征做标准化，所以是对最后一维进行操作，keepdim=True 是为了保持张量形状一致，以便后续广播
        var = x.var(dim=-1, keepdim=True, unbiased=False)   # 对最后一个维度（通常是 embedding 维度）求方差。unbiased=False：计算方差时用1/N,而不是1/(N-1),因为神经网络训练中通常需要有偏方差估计，不需要无偏估计
        norm_x = (x - mean) / torch.sqrt(var + self.eps)    # 标准化操作，让激活变为均值0、方差1。归一化有助于提高模型训练速度、稳定性，并减小梯度消失或爆炸的风险
        return self.scale * norm_x + self.shift # 应用可学习的缩放和偏移参数。恢复模型的表达能力，不受标准化的限制，可以学出更强的特征变换

if __name__ == "__main__":
    torch.manual_seed(123)  # 设置随机数种子，保证每次运行结果一致
    batch_example = torch.randn(2, 5)   # 创建一个随机的 2 x 5 的张量

    """
    构造一个顺序神经网络模块，按顺序执行以下操作：
    nn.Linear(5, 6)：全连接层.  输入维度为 5，输出维度为 6
    nn.ReLU()：激活函数（Rectified Linear Unit）,将负值变为 0，正值不变
    Input (5 dim) → Linear (5→6) → ReLU → Output (6 dim)
    """
    layer = nn.Sequential(nn.Linear(5, 6), nn.ReLU())

    """
    将输入的 batch_example（2×5 的张量）传入 layer，计算输出结果。
    实际执行顺序：
    输入 shape: [2, 5]
    通过 Linear(5, 6)，输出 shape: [2, 6]
    再经过 ReLU()，输出仍是 [2, 6]
    """
    out = layer(batch_example)

    # tensor([[0.2260, 0.3470, 0.0000, 0.2216, 0.0000, 0.0000],
    #         [0.2133, 0.2394, 0.0000, 0.5198, 0.3297, 0.0000]],
    #        grad_fn=<ReluBackward0>)
    print(out)

    ## 检查均值和方差，再进行层归一化操作
    """keepdim=False 表示不保留维度, keepdim=True 表示保留维度。dim=-1或者dim=1表示列维度，按最后一个维度求均值或方差（即行内求均值或方差）。dim=0表示行维度（即列内求均值或方差）。dim 指的是要“消除”（或者说压缩、聚合）哪个维度。
    对这个例子来说：
    keepdim=True，输入形状是 [2, 6]，所以 mean 和 var 的形状都是 [2, 1]。
    Mean:
    tensor([[0.1324],
            [0.2170]], grad_fn=<MeanBackward1>)
    Variance:
    tensor([[0.0231],
            [0.0398]], grad_fn=<VarBackward0>)
    keepdim=False，输入形状是 [2, 6]，所以  mean 和 var 的形状都是 [2]。也就是一个二维向量
    Mean:
    tensor([0.1324, 0.2170], grad_fn=<MeanBackward1>)
    Variance:
    tensor([0.0231, 0.0398], grad_fn=<VarBackward0>)"""
    mean = out.mean(dim=-1, keepdim=True)   # 计算每个样本的均值
    var = out.var(dim=-1, keepdim=True)     # 计算每个样本的方差

    print("Mean:\n", mean)
    print("Variance:\n", var)

    out_norm = (out - mean) / torch.sqrt(var)   # 对每个样本进行归一化

    ## 现在我们来验证它是否真的标准化了（均值 ≈ 0，方差 ≈ 1），验证结果正确
    mean = out_norm.mean(dim=-1, keepdim=True)
    var = out_norm.var(dim=-1, keepdim=True)

    """Normalized layer outputs:
    tensor([[ 0.6159,  1.4126, -0.8719,  0.5872, -0.8719, -0.8719],
            [-0.0189,  0.1121, -1.0876,  1.5173,  0.5647, -1.0876]],
        grad_fn=<DivBackward0>)"""
    print("Normalized layer outputs:\n", out_norm)
    torch.set_printoptions(sci_mode=False)  # 禁用科学计数法

    """Mean:
    tensor([[    0.0000],
            [    0.0000]], grad_fn=<MeanBackward1>)
    Variance:
    tensor([[1.0000],
            [1.0000]], grad_fn=<VarBackward0>)"""
    print("Mean:\n", mean)
    print("Variance:\n", var)


    ## 使用封装类
    ln = LayerNorm(emb_dim=5)
    out_ln = ln(batch_example)
    mean = out_ln.mean(dim=-1, keepdim=True)
    var = out_ln.var(dim=-1, unbiased=False, keepdim=True)
    print("Mean:\n", mean)
    print("Variance:\n", var)