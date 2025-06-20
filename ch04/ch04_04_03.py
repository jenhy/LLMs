import torch
import torch.nn as nn

"""
这是 Python 中常用的绘图库，用于生成图表。
我们需要绘制 GELU 和 ReLU 激活函数的图像，用 plt.plot、plt.title 等函数来生成图像。
"""
import matplotlib.pyplot as plt

class GELU(nn.Module):
    r"""
    GELU激活函数的实现
    """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(torch.sqrt(torch.tensor(2.0 / torch.pi)) * (x + 0.044715 * torch.pow(x, 3))))

class FeedForward(nn.Module):
    r"""
    使用GELU函数实现前馈神经网络,继承自nn.Module。通过继承 PyTorch 的基类 nn.Module，可以使用其构建模型的所有能力（如参数注册、自动微分等）。FeedForward 是 Transformer 中的关键组件之一（FFN层），封装成一个类可复用、便于管理模型结构。
    前馈网络层结构：x --> Linear(emb_dim → 4×emb_dim) --> GELU --> Linear(4×emb_dim → emb_dim)
    """
    def __init__(self, cfg):
        r"""
        定义构造函数，接收一个配置字典 cfg。cfg 是外部传入的配置，包含例如 emb_dim 等参数。让网络结构灵活可调，避免写死维度，适用于不同规模的模型。
        """

        r"""
        调用父类的构造函数 nn.Module.__init__()。通过 super() 自动找到父类并初始化。这是构建 PyTorch 模块的标准做法，确保模型可以注册参数、支持反向传播等功能。
        """
        super().__init__()

        r"""
        定义了一个顺序容器 self.layers，用于串联多个子层。nn.Sequential 是 PyTorch 提供的模块组合器，按顺序调用其中的层。可以让前向传播结构清晰简洁，方便构建标准前馈结构。
        第一个全连接层（线性层），输入维度为 emb_dim，输出维度扩大为 4 倍。nn.Linear(in_dim, out_dim) 实现线性变换 y=Wx+b。这是 Transformer FFN 的标准结构，先升维扩展特征空间以增加非线性表达能力。
        应用 GELU 激活函数（高斯误差线性单元）。GELU 是非线性激活函数，输出更平滑、对梯度更友好。相比 ReLU 更平滑、拟合能力更强，是 BERT 和 GPT 等大型模型推荐的激活函数。
        第二个线性层，把特征维度从 4 倍压缩回原始的 emb_dim。执行矩阵乘法，实现从高维特征空间回投。构成完整的 FFN（前馈网络）结构：升维 → 激活 → 降维，这是深度学习中提取更复杂特征的常见模式。
        """
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"])
        )

    def forward(self, x):
        r"""
        定义前向传播函数。重写 nn.Module 的 forward() 方法，定义输入张量经过本模块的处理逻辑。PyTorch 通过调用 model(input) 自动执行 forward()，这是自定义模块的必要步骤。
        """

        r"""
        将输入 x 依次通过 self.layers 中的所有层。调用 nn.Sequential 对象，相当于按顺序执行三层操作。这样封装可以让 FeedForward 变成一个可插拔的网络模块，易于集成进更大的 Transformer 框架中。
        """
        return self.layers(x)

if __name__ == "__main__":
    """
    创建两个激活函数实例：GELU（高斯误差线性单元Gaussian Error Linear Unit）和 ReLU（修正线性单元）。
    我们要比较这两种常见的激活函数在输入区间 [-3, 3] 上的输出行为，因此需要先实例化它们。
    """
    gelu, relu = GELU(), nn.ReLU()

    """
    创建一个张量 x，在区间 [-3, 3] 上均匀采样 100 个点。
    用于作为激活函数的输入横坐标，这样我们可以画出激活函数在这个区间内的曲线。
    """
    x = torch.linspace(-3, 3, 100)

    """
    分别对 x 应用 gelu 和 relu 激活函数，得到对应的输出 y_gelu 和 y_relu。
    这是为了可视化这两个函数的输出形状，即曲线图上的纵坐标值。
    """
    y_gelu, y_relu = gelu(x), relu(x)

    """
    创建一个图形窗口，大小为 8 英寸宽、3 英寸高。
    设置图像显示的尺寸，使得两个子图可以在水平方向上并排展示。
    """
    plt.figure(figsize=(8, 3))

    """
    使用 zip 将两个输出和标签配对，用 enumerate(..., 1) 给每个组合编号（从1开始）。
    方便在接下来的 subplot 中指定第几个子图，并为每个子图添加标签。
    """
    for i, (y, label) in enumerate(zip([y_gelu, y_relu], ["GELU", "ReLU"]), 1):
        # 在一行两列的图中选中第 i 个子图。要将两个激活函数的曲线分别画在两个子图上，方便对比。
        plt.subplot(1, 2, i)
        # 画出 x 与 y 的关系曲线。这是激活函数的图像，是这段代码的核心目标。
        plt.plot(x, y)
        # 为当前子图设置标题。告诉我们当前图画的是哪个激活函数。
        plt.title(f"{label} activation function")
        # 为 x 轴设置标签。标明横轴是输入 x。
        plt.xlabel("x")
        # 为 y 轴设置标签，表示这是激活函数对 x 的输出。
        plt.ylabel(f"{label}(x)")
        # 启用网格线。让图像更易读，特别是对比曲线变化时更清晰。
        plt.grid(True)
    plt.tight_layout()  # 自动调整子图参数，使图像布局更紧凑。避免图像内容重叠，提高美观和可读性。
    plt.show()  # 显示整个图像窗口。这是 matplotlib 的标准用法，用来展示生成的图形。


    """
    ReLU 会在 0 左右发生剧烈“断层”（x<0 输出为 0）。GELU 是平滑的 S 形曲线，提供更平滑的梯度，训练更稳定。
    """

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

    ffn = FeedForward(GPT_CONFIG_124M)
    x = torch.rand(2, 3, 768)
    out = ffn(x)
    print(out.shape)    # torch.Size([2, 3, 768])