import torch
from torch.utils.data import Dataset

import os, sys
# 添加上级路径以便导入自定义模块
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)  
sys.path.append(parent_dir)

from ch07.ch07_07_02 import format_input

class InstructionDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.encoded_texts = []
        for entry in data:
            instruction_plus_input = format_input(entry)
            response_text = f"\n\n### Response:\n{entry['output']}"
            full_text = instruction_plus_input + response_text
            self.encoded_texts.append(tokenizer.encode(full_text))

    def __getitem__(self, index):
        return self.encoded_texts[index]
    
    def __len__(self):
        return len(self.data)
    
def custom_collate_draft_1(batch, pad_token_id=50256, device="cpu"):
    """
    填充每一个批次。
    """

    # 获取批次中的最大长度。len(item) + 1 是为了给每个样本加上终止符号。
    batch_max_length = max(len(item)+1 for item in batch)
    # print(f"batch_max_length: {batch_max_length}")
    inputs_lst = []

    for item in batch:
        # 防止修改原始 item 内容
        new_item = item.copy()
        new_item += [pad_token_id]
        # print(f"new_item: {new_item}")

        padded = (new_item + [pad_token_id] * (batch_max_length - len(new_item)))
        # print(f"padded: {padded}")
        # print(f"padded[:-1]: {padded[:-1]}")

        # 把之前末尾额外填充的词元删除掉，作为“输入序列”，这样模型预测的是“下一个 token”
        inputs = torch.tensor(padded[:-1])
        inputs_lst.append(inputs)
        # print(f"inputs: {inputs}")
    # print(f"inputs_lst: {inputs_lst}")

    """
    输入列表变成一个张量。
    如：
    inputs_lst: [tensor([0, 1, 2, 3, 4]), tensor([    5,     6, 50256, 50256, 50256]), tensor([    7,     8,     9, 50256, 50256])]
    tensor([[    0,     1,     2,     3,     4],
            [    5,     6, 50256, 50256, 50256],
            [    7,     8,     9, 50256, 50256]])
    """
    input_tensors = torch.stack(inputs_lst).to(device)
    return input_tensors

def custom_collate_draft_2(batch, pad_token_id=50256, device="cpu"):
    # 获取批次中的最大长度。len(item) + 1 是为了给每个样本加上终止符号。
    batch_max_length = max(len(item)+1 for item in batch)
    inputs_lst, targets_lst = [], []
    for item in batch:
        # 防止修改原始 item 内容
        new_item = item.copy()
        new_item += [pad_token_id]
        padded = (new_item + [pad_token_id] * (batch_max_length - len(new_item)))
        inputs = torch.tensor(padded[:-1])
        # 截断输入的最后一个词元
        targets = torch.tensor(padded[1:])
        inputs_lst.append(inputs)
        # 向左移动一个位置得到目标
        targets_lst.append(targets)
    
    inputs_tensor = torch.stack(inputs_lst).to(device)
    targets_tensor = torch.stack(targets_lst).to(device)
    return inputs_tensor, targets_tensor

def custom_collate_fn(batch, pad_token_id=50256, ignore_index=-100, allowed_max_length=None, device="cpu"):
    # 获取批次中的最大长度。len(item) + 1 是为了给每个样本加上终止符号。
    batch_max_length = max(len(item)+1 for item in batch)
    inputs_lst, targets_lst = [], []
    for item in batch:
        # 防止修改原始 item 内容
        new_item = item.copy()
        new_item += [pad_token_id]
        padded = (new_item + [pad_token_id] * (batch_max_length - len(new_item)))
        inputs = torch.tensor(padded[:-1])

        # 截断输入的最后一个词元
        targets = torch.tensor(padded[1:])

        # 布尔张量。表示 targets 中哪些位置是 pad_token_id。使用 == 运算符进行逐元素比较，返回 True/False
        # 如：targets = tensor([10, 20, 50256, 50256, 50256])，pad_token_id = 50256，那么mask = tensor([False, False, True, True, True])
        mask = targets == pad_token_id
        
        # indices 表示 pad_token_id 出现位置的索引张量，也可以理解为布尔 mask 中为 True 的位置下标。
        # torch.nonzero(mask) 返回的是二维张量，每一行是一个 index。如：tensor([[2],[3],[4]])。
        # .squeeze() 会将这个二维张量降维成一维。如：tensor([2, 3, 4])
        # 如：targets = tensor([10, 20, 50256, 50256, 50256])，pad_token_id = 50256，那么indices = tensor([2, 3, 4])，也就是索引张量。
        indices = torch.nonzero(mask).squeeze()

        # 判断是否找到了两个或者以上的填充位置
        if indices.numel() > 1:

            # 将除第一个以外的所有 padding 位置上的目标 token 设置为 ignore_index（默认是 -100，PyTorch 的 CrossEntropyLoss 会忽略这个值）。
            # indices[1:] 取的是除第一个 padding 位置以外的其余位置。
            targets[indices[1:]] = ignore_index

        if allowed_max_length is not None:
            inputs = inputs[:allowed_max_length]
            targets = targets[:allowed_max_length]

        inputs_lst.append(inputs)
        # 向左移动一个位置得到目标
        targets_lst.append(targets)
    
    inputs_tensor = torch.stack(inputs_lst).to(device)
    targets_tensor = torch.stack(targets_lst).to(device)
    return inputs_tensor, targets_tensor


if __name__ == "__main__":
    inputs_1 = [0, 1, 2, 3, 4]
    inputs_2 = [5, 6]
    inputs_3 = [7, 8, 9]

    """
    custom_collate_draft_1:
    tensor([[    0,     1,     2,     3,     4],
            [    5,     6, 50256, 50256, 50256],
            [    7,     8,     9, 50256, 50256]])
    """
    batch = (inputs_1, inputs_2, inputs_3)
    print("custom_collate_draft_1:")
    print(custom_collate_draft_1(batch))

    """
    custom_collate_draft_2:
    tensor([[    0,     1,     2,     3,     4],
            [    5,     6, 50256, 50256, 50256],
            [    7,     8,     9, 50256, 50256]])
    tensor([[    1,     2,     3,     4, 50256],
            [    6, 50256, 50256, 50256, 50256],
            [    8,     9, 50256, 50256, 50256]])
    """
    inputs, targets = custom_collate_draft_2(batch)
    print("custom_collate_draft_2:")
    print(inputs)
    print(targets)

    """
    custom_collate_fn:
    tensor([[    0,     1,     2,     3,     4],
            [    5,     6, 50256, 50256, 50256],
            [    7,     8,     9, 50256, 50256]])
    tensor([[    1,     2,     3,     4, 50256],
            [    6, 50256,  -100,  -100,  -100],
            [    8,     9, 50256,  -100,  -100]])
    """
    inputs, targets = custom_collate_fn(batch)
    print("custom_collate_fn:")
    print(inputs)
    print(targets)

    logits_1 = torch.tensor([[-1.0, 1.0],
                  [-0.5, 1.5]]
    )
    targets_1 = torch.tensor([0, 1])
    loss_1 = torch.nn.functional.cross_entropy(logits_1, targets_1)
    print("loss_1:", loss_1)

    logits_2 = torch.tensor([[-1.0, 1.0],
                  [-0.5, 1.5],
                  [-0.5, 1.5]]
    )
    targets_2 = torch.tensor([0, 1, 1])
    loss_2 = torch.nn.functional.cross_entropy(logits_2, targets_2)
    print("loss_2:", loss_2)

    logits_3 = torch.tensor([[-1.0, 1.0],
                  [-0.5, 1.5],
                  [-0.5, 1.5]]
    )
    targets_3 = torch.tensor([0, 1, -100])
    loss_3 = torch.nn.functional.cross_entropy(logits_3, targets_3)
    print("loss_3:", loss_3)
    print("loss_1 == loss_3:", loss_1 == loss_3)