import torch
import torch.nn.functional as F #提供了各种函数式的神经网络组件，比如损失函数。
from torch.autograd import grad #用于显式计算梯度（不像 .backward() 会自动累加到 .grad 上）。

y = torch.tensor([1.0]) #定义标签。 y 是标签值，即真实值。这里是二分类问题中正类的标签：1。
x1 = torch.tensor([1.1])    #定义输入特征。这是模型的输入特征值1.1
w1 = torch.tensor([2.2], requires_grad=True)    #定义权重参数。requires_grad=True 表示这个张量需要计算梯度（也就是说这是一个可训练参数）。
b = torch.tensor([0.0], requires_grad=True) #定义偏置项b。同样是一个需要求梯度的参数。

z = x1 * w1 + b #线性变换
a = torch.sigmoid(z)    # sigmoid激活函数

print(z)    # tensor([2.4200]
print(a)    # tensor([0.9183]

#计算二分类交叉熵损失函数，用来衡量模型预测a和真实标签y之间的差距。loss越小越好（越接近 0）
#预测值a=1.0（完全正确），log(1)=0  -->  loss=0
#如何让loss变小？通过调整w1和b参数。x1是输入特征，w1是权重参数，b是偏置项。x1通常是不变的(训练集数据给的)，w1和b是可训练参数，是我们训练的目标，可以调整。
loss = F.binary_cross_entropy(a, y)

print(loss) # tensor(0.0852, grad_fn=<BinaryCrossEntropyBackward0>) 可以看到输出有 grad_fn，说明它是计算图中的一部分，支持反向传播。

## 方法1：手动计算梯度，而不是使用 .backward()
# retain_graph=True 保留计算图（否则 PyTorch 默认在计算一次后会释放资源）
grad_L_w1 = grad(loss, w1, retain_graph=True)
grad_L_b = grad(loss, b, retain_graph=True)

# grad() 显示在 tuple 里，输出的是元组
# (tensor([-0.0898]),)
# (tensor([-0.0817]),)
print(grad_L_w1)
print(grad_L_b)

## 方法2：使用.backward()
# 计算梯度，并累加到 .grad 上
# 自动反向传播并将梯度存到参数的.grad属性中，通过 w1.grad 和 b.grad 来获取梯度
loss.backward()

# tensor([-0.0898])
# tensor([-0.0817])
print(w1.grad)
print(b.grad)