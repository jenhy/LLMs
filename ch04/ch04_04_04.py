import torch
import torch.nn as nn
from ch04_04_03 import GELU

class ExampleDeepNeuralNetwork(nn.Module):
    r"""
    定义一个继承自 nn.Module 的深度神经网络类。通过继承 nn.Module，你可以重写构造函数 __init__ 和前向传播函数 forward。PyTorch 所有模型都需要继承 nn.Module 以获得自动参数注册、模型保存等功能。
    """
    def __init__(self, layer_sizes, use_shortcut):
        r"""
        初始化函数，接收层尺寸和是否使用残差连接作为参数。layer_sizes 表示网络每层的维度，use_shortcut 控制是否使用快捷/残差/跳跃连接。
        """
        super().__init__()  # 调用父类的构造函数super().__init__() 是必须的，才能正确注册模块。
        self.use_shortcut = use_shortcut    # 保存是否使用残差连接的配置。残差连接可以缓解深层网络中的梯度消失/退化问题（如 ResNet 所提出的）。

        r"""
        创建一个由 5 个子模块组成的网络，每个子模块是一个 Sequential 层。
        每一层是 nn.Linear + GELU() 的组合。
        PyTorch 的 nn.Linear(in_features, out_features) 的作用是：接收形状为 [batch_size, in_features] 的输入，输出形状为 [batch_size, out_features] 的结果。
        """
        self.layers = nn.ModuleList(
            [
                nn.Sequential(nn.Linear(layer_sizes[0], layer_sizes[1]), GELU()),
                nn.Sequential(nn.Linear(layer_sizes[1], layer_sizes[2]), GELU()),
                nn.Sequential(nn.Linear(layer_sizes[2], layer_sizes[3]), GELU()),
                nn.Sequential(nn.Linear(layer_sizes[3], layer_sizes[4]), GELU()),
                nn.Sequential(nn.Linear(layer_sizes[4], layer_sizes[5]), GELU()),
            ]
        )

    def forward(self, x):
        r"""
        前向传播部分
        PyTorch 模型必须实现 forward 方法来定义数据流逻辑。
        """

        r"""
        遍历每一层并执行计算。
        layer(x) 调用 Sequential(Linear + GELU)。逐层构建神经网络前向传播路径。
        """
        for layer in self.layers:
            layer_output = layer(x) # 计算当前层的输出，形状：[batch_size, seq_len, hidden_size]

            r"""
            实现残差连接。
            如果启用残差连接且当前层输出的形状和输入形状相同，则执行 x + layer_output。只有当输入输出形状相同（维度未变）才适合直接加。
            残差连接能使训练更深层网络更稳定。
            """
            if self.use_shortcut and x.shape == layer_output.shape:
                x = x + layer_output
            else:
                x = layer_output

        return x    # 返回最后一层的输出。

def print_gradients(model, x):
    r"""
    这个函数用于测试和可视化模型中参数的梯度值。
    """
    output = model(x)   # 前向传播得到输出。
    target = torch.tensor([[0.]], dtype=output.dtype)   # 设定一个简单的目标输出（标签）。形状为 [1, 1]。

    # 构造损失函数并计算损失。MSELoss是常用的回归损失。
    loss = nn.MSELoss()
    loss = loss(output, target)

    # 反向传播，计算梯度。只有调用.backward() 后，PyTorch 会追踪各层参数的导数。
    loss.backward()

    r"""
    遍历所有参数，打印权重的梯度平均绝对值。
    帮助判断是否有梯度消失或爆炸问题（梯度为0或特别大）。
    """
    for name, param in model.named_parameters():
        # print(f"{name}: {param.grad.abs().mean()}")
        if 'weight' in name:
            print(f"{name} has gradient mean of {param.grad.abs().mean().item()}")

if __name__ == "__main__":
    layer_sizes = [3, 3, 3, 3, 3, 1]    #   定义每层的输入输出维度。
    sample_input = torch.tensor([[1., 0., -1.]])    #   提供一组样本输入，形状为 [1, 3]。表示一个 batch、三维向量的输入样本。匹配第一层 in_features=3。第一层的输入维度layer_sizes[0]与输入样本sample_input必须匹配
    torch.manual_seed(123)  # 设置随机种子，保证每次运行结果相同。
    model_without_shortcut = ExampleDeepNeuralNetwork(layer_sizes, use_shortcut=False)  #   初始化一个不使用残差连接的实例模型。
    """    ExampleDeepNeuralNetwork(
    (layers): ModuleList(
        (0-3): 4 x Sequential(
        (0): Linear(in_features=3, out_features=3, bias=True)
        )
        (4): Sequential(
        (0): Linear(in_features=3, out_features=1, bias=True)
        )
    )
    )"""
    # print(model_without_shortcut)
    output = model_without_shortcut(sample_input)  # 执行前向传播
    # print(output)

    """
    从打印结果看，每一层的梯度非常小，越往后层越大（尤其是最后一层），但整体依然很小，暗示前几层几乎学不到东西。梯度从最后一层layers.4到第一层layers.0的过程逐渐变小，这种现象称为【梯度消失】。
    layers.0.0.weight has gradient mean of 0.00020173587836325169
    layers.1.0.weight has gradient mean of 0.0001201116101583466
    layers.2.0.weight has gradient mean of 0.0007152041653171182
    layers.3.0.weight has gradient mean of 0.001398873864673078
    layers.4.0.weight has gradient mean of 0.005049646366387606
    """
    print_gradients(model_without_shortcut, sample_input)   # 调用梯度检查函数。


    # 初始化一个使用残差连接的实例模型。
    torch.manual_seed(123)
    model_with_shortcut = ExampleDeepNeuralNetwork(layer_sizes, use_shortcut=True)
    """
    从打印结果看，各层梯度均值都比较大，且逐层分布合理 —— 梯度能有效地传递到每一层，有利于深层网络的训练。
    layers.0.0.weight has gradient mean of 0.22169792652130127
    layers.1.0.weight has gradient mean of 0.20694106817245483
    layers.2.0.weight has gradient mean of 0.32896995544433594
    layers.3.0.weight has gradient mean of 0.2665732502937317
    layers.4.0.weight has gradient mean of 1.3258541822433472
    """
    print_gradients(model_with_shortcut, sample_input)



r"""
调试过程中遇到的问题：
PS C:\Users\Jenhy\OneDrive\doc\学习\AI\LLMs> & "D:/Program Files/Python/Python38/python.exe" c:/Users/Jenhy/OneDrive/doc/学习/AI/LLMs/ch04/ch04_04_04.py
Traceback (most recent call last):
  File "c:/Users/Jenhy/OneDrive/doc/学习/AI/LLMs/ch04/ch04_04_04.py", line 59, in <module>
    print_gradients(model_without_shortcut, sample_input)
  File "c:/Users/Jenhy/OneDrive/doc/学习/AI/LLMs/ch04/ch04_04_04.py", line 32, in print_gradients
    loss = loss(output, target)
  File "D:\Program Files\Python\Python38\lib\site-packages\torch\nn\modules\module.py", line 1553, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "D:\Program Files\Python\Python38\lib\site-packages\torch\nn\modules\module.py", line 1562, in _call_impl
    return forward_call(*args, **kwargs)
  File "D:\Program Files\Python\Python38\lib\site-packages\torch\nn\modules\loss.py", line 538, in forward
    return F.mse_loss(input, target, reduction=self.reduction)
  File "D:\Program Files\Python\Python38\lib\site-packages\torch\nn\functional.py", line 3373, in mse_loss
    if not (target.size() == input.size()):
AttributeError: 'NoneType' object has no attribute 'size'

问题原因：
忘了在 forward 方法的最后一行添加 return x，所以默认返回了 None，导致 output = model(x) 的结果是 None。

"""