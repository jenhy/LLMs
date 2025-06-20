import torch
import torch.nn as nn
from ch03_05_03 import CausalAttention

"""
MultiHeadAttentionWrapper是一个“包装器”类，它把多个 CausalAttention 实例（也就是多个注意力头）组合成一个整体。
"""
class MultiHeadAttentionWrapper(nn.Module):
    
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        """
        nn.ModuleList是一个用来注册多个子模块的容器。把多个注意力头模块放进 nn.ModuleList，注册为子模块的一部分
        注意：不能这样写：self.heads = [CausalAttention(...), CausalAttention(...)]
        PyTorch 不会自动发现这两个子模块，也就无法注册这些模块的参数
        """
        self.heads = nn.ModuleList(
            [
                CausalAttention(d_in, d_out, context_length, dropout, qkv_bias)
                for _ in range(num_heads)
            ]
        )

    def forward(self, x):
        # 遍历 self.heads 中的每一个注意力头 head，把输入 x 喂给它，收集所有头的输出结果，最后在最后一个维度 dim=-1 上拼接。
        # 当写 head(x) 的时候，其实就是在执行：head.__call__(x) → 它自动转发到 → head.forward(x)
        # 为什么要用 torch.cat(..., dim=-1) 来拼接多个注意力头的结果？因为 “多头注意力”的设计初衷，就是让每个注意力头专注于不同的子空间，然后把这些子空间的结果拼接在一起，组合成更丰富、更强大的上下文表示。
        return torch.cat([head(x) for head in self.heads], dim=-1)
    
if __name__ == '__main__':
    inputs = torch.tensor([[0.43, 0.15, 0.89],
                        [0.55, 0.87, 0.66],
                        [0.57, 0.85, 0.64],
                        [0.22, 0.58, 0.33],
                        [0.77, 0.25, 0.10],
                        [0.05, 0.80, 0.55]])

    """
    torch.stack 这个函数名中的 "stack" 是英文单词，意思是 “堆叠”、“叠放”，它的功能也正如其名：将多个张量沿着一个新维度“堆叠”在一起，生成一个新的更高维的张量。
    将两个形状为 (6, 3) 的张量沿着新维度（第 0 维）堆叠成一个形状为 (2, 6, 3) 的三维张量。
    第 0 维：表示有2个“输入样本集”(可以理解为 batch size = 2)
    第 1 维：每个样本集有 6 个样本
    第 2 维：每个样本有 3 个特征
    """
    batch = torch.stack((inputs, inputs), dim=0)
    # print(batch)
    # batch.shape[0]:2,batch.shape[1]:6,batch.shape[2]:3
    # print(f"batch.shape[0]:{batch.shape[0]},batch.shape[1]:{batch.shape[1]},batch.shape[2]:{batch.shape[2]}")
    d_in = inputs.shape[1] # 输入嵌入维度3
    d_out = 2 # 输出嵌入维度2，如果改成1，输出上下文向量就是2维

    torch.manual_seed(123)
    context_length = batch.shape[1] # 词元的数量6
    mha = MultiHeadAttentionWrapper(d_in, d_out, context_length, 0.0, num_heads=2)
    """
    MultiHeadAttentionWrapper(
    (heads): ModuleList(
        (0-1): 2 x CausalAttention(
        (W_query): Linear(in_features=3, out_features=2, bias=False)
        (W_key): Linear(in_features=3, out_features=2, bias=False)
        (W_value): Linear(in_features=3, out_features=2, bias=False)
        (dropout): Dropout(p=0.0, inplace=False)
        )
    )
    )
    """
    print(mha)  # 会打印 ModuleList 中注册了多个注意力头
    """
    [Parameter containing:
    tensor([[-0.2354,  0.0191, -0.2867],
            [ 0.2177, -0.4919,  0.4232]], requires_grad=True), Parameter containing:
    tensor([[-0.4196, -0.4590, -0.3648],
            [ 0.2615, -0.2133,  0.2161]], requires_grad=True), Parameter containing:
    tensor([[-0.4900, -0.3503, -0.2120],
            [-0.1135, -0.4404,  0.3780]], requires_grad=True), Parameter containing:
    tensor([[-0.1362,  0.1853,  0.4083],
            [ 0.1076,  0.1579,  0.5573]], requires_grad=True), Parameter containing:
    tensor([[-0.2604,  0.1829, -0.2569],
            [ 0.4126,  0.4611, -0.5323]], requires_grad=True), Parameter containing:
    tensor([[ 0.4929,  0.2757,  0.2516],
        [ 0.2377,  0.4800, -0.0762]], requires_grad=True)]
    """
    print(list(mha.parameters()))   # 输出每个头的所有参数
    context_vecs = mha(batch)
    """
    tensor([[[-0.4519,  0.2216,  0.4772,  0.1063],
         [-0.5874,  0.0058,  0.5891,  0.3257],
         [-0.6300, -0.0632,  0.6202,  0.3860],
         [-0.5675, -0.0843,  0.5478,  0.3589],
         [-0.5526, -0.0981,  0.5321,  0.3428],
         [-0.5299, -0.1081,  0.5077,  0.3493]],

        [[-0.4519,  0.2216,  0.4772,  0.1063],
         [-0.5874,  0.0058,  0.5891,  0.3257],
         [-0.6300, -0.0632,  0.6202,  0.3860],
         [-0.5675, -0.0843,  0.5478,  0.3589],
         [-0.5526, -0.0981,  0.5321,  0.3428],
         [-0.5299, -0.1081,  0.5077,  0.3493]]], grad_fn=<CatBackward0>)
    """
    print(context_vecs)
    print("context_vecs.shape:", context_vecs.shape)    # torch.Size([2, 6, 4]) 。这是执行torch.cat拼接操作后的结果，即(2, 6, 2)拼接(2, 6, 2)最后一维的结果