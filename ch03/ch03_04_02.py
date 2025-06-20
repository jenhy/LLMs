import torch
import torch.nn as nn   # 构建神经网络netural network的模块



f"""
总体流程：
输入 x
  ├─> Q = x @ W_query ─┐
  ├─> K = x @ W_key    ├─> Q @ K.T → attn_score → softmax → attn_weight
  └─> V = x @ W_value  ┘                               ↓
                                                  attn_weight @ V
                                                       ↓
                                                    context"""


class SelfAttentionV1(nn.Module):
    """
    使用 nn.Parameter 手动定义权重。不支持偏置
    """
    def __init__(self, d_in, d_out):
        """
        d_in: 输入向量的维度
        d_out: 输出向量(Query/Key/Value)的维度
        """
        super().__init__()
        # 用来生成Query（查询）、Key（键）、Value（值）向量的权重矩阵。
        self.W_query = nn.Parameter(torch.rand(d_in, d_out))
        self.W_key = nn.Parameter(torch.rand(d_in, d_out))  
        self.W_value = nn.Parameter(torch.rand(d_in, d_out))
        print(f"self.W_query.shape:", self.W_query.shape)       # 输出形状为：3 * 2 即：self.W_query.shape: torch.Size([3, 2])
    def forward(self, x):
        """
        前向传播
        x: 输入张量,形状为(N,d_in)
        """

        # 1、计算 Query、Key、Value 向量
        # @ 是矩阵乘法。将每个输入向量投影到新的空间中。输出形状为(N,d_out)
        queries = x @ self.W_query
        keys = x @ self.W_key
        values = x @ self.W_value

        # 2、计算注意力分数
        # queries: (N, d_out),keys.T: (d_out, N),输出形状为(N, N)
        attn_score = queries @ keys.T

        # 3、缩放（Scaling）+ Softmax 归一化
        # shape[-1] 是 Python 和 PyTorch 中的一种“倒数索引”写法，意思是 “取张量最后一个维度的大小”。也就是d_out
        # attn_weight: (N, N)，注意力权重矩阵。
        attn_weight = torch.softmax(attn_score / keys.shape[-1] ** 0.5, dim=-1)

        # 4、计算上下文向量（Context Vector）
        # attn_weight: (N, N),values: (N, d_out),输出形状为(N,d_out)
        context = attn_weight @ values
        return context

class SelfAttentionV2(nn.Module):
    """
    使用 nn.Linear 层定义权重，支持偏置，通过qkv_bias参数控制是否使用偏置。
    """
    def __init__(self, d_in, d_out, qkv_bias=False):
        """
        d_in: 输入向量的维度
        d_out: 输出向量(Query/Key/Value)的维度
        qkv_bias: 是否使用偏置
        """
        super().__init__()
        # 用来生成Query（查询）、Key（键）、Value（值）向量的权重矩阵。
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)  
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        print(f"self.W_query.weight.shape:", self.W_query.weight.shape) # 输出形状为：2 * 3，即self.W_query.weight.shape: torch.Size([2, 3])
        
        """
        通过.weight属性，可以查看权重矩阵。
        self.W_query.weight: Parameter containing:
        tensor([[ 0.3161,  0.4568,  0.5118],
        [-0.1683, -0.3379, -0.0918]], requires_grad=True)
        """
        print(f"self.W_query.weight:", self.W_query.weight)
    def forward(self, x):
        """
        前向传播
        x: 输入张量,形状为(N,d_in)
        """

        # 1、计算 Query、Key、Value 向量
        # @ 是矩阵乘法。将每个输入向量投影到新的空间中。输出形状为(N,d_out)
        queries = self.W_query(x)
        keys = self.W_key(x)
        values = self.W_value(x)

        # 2、计算注意力分数
        # queries: (N, d_out),keys.T: (d_out, N),输出形状为(N, N)
        attn_score = queries @ keys.T

        # 3、缩放（Scaling）+ Softmax 归一化
        # shape[-1] 是 Python 和 PyTorch 中的一种“倒数索引”写法，意思是 “取张量最后一个维度的大小”。也就是d_out
        # attn_weight: (N, N)，注意力权重矩阵。
        attn_weight = torch.softmax(attn_score / keys.shape[-1] ** 0.5, dim=-1)

        # 4、计算上下文向量（Context Vector）
        # attn_weight: (N, N),values: (N, d_out),输出形状为(N,d_out)
        context = attn_weight @ values
        return context
    
if __name__ == '__main__':
    inputs = torch.tensor([[0.43, 0.15, 0.89],
                       [0.55, 0.87, 0.66],
                       [0.57, 0.85, 0.64],
                       [0.22, 0.58, 0.33],
                       [0.77, 0.25, 0.10],
                       [0.05, 0.80, 0.55]])
    x_2 = inputs[1] # x_2形状为：3
    print(x_2.shape)
    d_in = inputs.shape[1] # 输入嵌入维度3
    d_out = 2 # 输出嵌入维度2

    ## 使用手动实现
    torch.manual_seed(123) # 固定设置 PyTorch 的随机数种子（seed）为 123，以确保后续生成的随机数是“可复现的”。
    sa_v1 = SelfAttentionV1(d_in, d_out)
    # 输入inputs包含6个嵌入向量,输出为：6个2维的嵌入向量
    # tensor([[0.2996, 0.8053],
    #     [0.3061, 0.8210],
    #     [0.3058, 0.8203],
    #     [0.2948, 0.7939],
    #     [0.2927, 0.7891],
    #     [0.2990, 0.8040]], grad_fn=<MmBackward0>)
    print(sa_v1(inputs))    


    """
    关于sa_v1(inputs)用法的解释：
    在 PyTorch 中，nn.Module 的子类（你自定义的 SelfAttentionV1）重载了 __call__ 方法，当你调用 sa_v1(inputs) 时，其实是隐式调用了：sa_v1.__call__(inputs)。而 __call__() 方法内部会自动调用你的 forward() 方法。所以sa_v1(inputs) === sa_v1.__call__(inputs) === sa_v1.forward(inputs)（带额外功能）

    为什么不直接调用 sa_v1.forward(inputs)，而要通过 sa_v1(inputs) 这种“间接方式”？
    用 sa_v1(inputs) 是更 专业、通用、可扩展、安全 的方式。用 sa_v1.forward(inputs) 虽然“看起来直观”，但 绕过了 PyTorch 的关键机制，容易出错且不推荐。
    | 特性         | `sa_v1(inputs)`（推荐）           | `sa_v1.forward(inputs)`（不推荐） |
    | ---------- | ----------------------------- | ---------------------------- |
    | 是否调用 hooks | ✅ 会触发 forward/backward hooks  | ❌ 不会触发任何 hook                |
    | 是否跟踪模型状态   | ✅ 会识别 `train()` / `eval()` 模式 | ❌ 不会识别                       |
    | 是否构建计算图    | ✅ 正确构建计算图                     | ✅ 会构建，但更容易出错                 |
    | 是否统一调用方式   | ✅ 模型模块统一接口                    | ❌ 非通用写法                      |
    | 官方推荐       | ✅ 是 PyTorch 官方标准调用方式          | ❌ 被视为内部实现细节，不建议直接用           |
    """

    ## 使用nn.Linear实现

    torch.manual_seed(789)
    sa_v2 = SelfAttentionV2(d_in, d_out)
    # 输入inputs包含6个嵌入向量,输出为：6个2维的嵌入向量
    # tensor([[-0.0739,  0.0713],
    #         [-0.0748,  0.0703],
    #         [-0.0749,  0.0702],
    #         [-0.0760,  0.0685],
    #         [-0.0763,  0.0679],
    #         [-0.0754,  0.0693]], grad_fn=<MmBackward0>)
    print(sa_v2(inputs))

    """
    同样的输入张量，为什么使用手工实现的 SelfAttentionV1 和使用 nn.Linear 实现的 SelfAttentionV2，得到的输出张量不一样？
    因为它们的权重初始化方式不同！
    SelfAttentionV1 使用了手动随机初始化权重：torch.rand(...)（均匀分布 [0, 1]）
    SelfAttentionV2 使用了 nn.Linear，它内部默认用的是更适合训练的初始化方式（如 Kaiming Uniform）
    即使你设置了相同的 torch.manual_seed()，这两种方式产生的权重也不会相同，因此最终输出也不同。
    注意：nn.Linear 以转置的形式存储权重矩阵，因此，nn.Linear 的权重矩阵的形状为 (d_out, d_in)，而 SelfAttentionV1 的权重矩阵的形状为 (d_in, d_out)。
    """
    # tensor([[-0.0739,  0.0713],
    #         [-0.0748,  0.0703],
    #         [-0.0749,  0.0702],
    #         [-0.0760,  0.0685],
    #         [-0.0763,  0.0679],
    #         [-0.0754,  0.0693]], grad_fn=<MmBackward0>)
    sa_v1.W_query = torch.nn.Parameter(sa_v2.W_query.weight.T)
    sa_v1.W_key = torch.nn.Parameter(sa_v2.W_key.weight.T)
    sa_v1.W_value = torch.nn.Parameter(sa_v2.W_value.weight.T)
    print(sa_v1(inputs))    
