import torch
from torch.utils.data import Dataset, DataLoader
import tiktoken

class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        token_ids = tokenizer.encode(txt)   # 对全部文本进行分词

        # 使用滑动窗口将文本划分为长度为max_length的重叠序列，并保存到input_ids和target_ids中
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1:i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        """
        返回数据集的总行数
        """
        return len(self.input_ids)

    def __getitem__(self, idx):
        """
        返回数据集的指定行的长度
        """
        return self.input_ids[idx], self.target_ids[idx]
    

def create_dataloader_v1(txt, batch_size=4, max_length=256, stride=128, shuffle=True, drop_last=True, num_workers=0):
    """
    batch_size: 批量大小，表示每个batch中包含多少个训练样本。如batch_size等于4表示每批的训练样本有4个。
    num_workers: 多线程加载数据的线程数
    max_length: 最大长度
    stride: 滑动窗口的步长。每条数据之间有重叠，用了滑动窗口。当stride=max_length时，相当于没有重叠。然后把这些样本按batch_size为一组打包成 DataLoader。
    shuffle：是否打乱数据集，True表示打乱数据集，False表示不打乱数据集。
    drop_last: 是否丢弃最后一批数据，True表示如果最后一批数据长度不足batch_size个训练样本，则丢弃。
    """
    tokenizer = tiktoken.get_encoding("gpt2")   # 初始化分词器
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers)
    return dataloader

if __name__ == "__main__":
    with open("the-verdict.txt", "r", encoding="utf-8") as f:
        raw_text = f.read()
    print("原始文本长度:", len(raw_text))

    dataloader = create_dataloader_v1(raw_text, batch_size=8, max_length=4, stride=4, shuffle=False)
    data_iter = iter(dataloader)
    first_batch = next(data_iter)
    print(first_batch)  # [tensor([[  40,  367, 2885, 1464]]), tensor([[ 367, 2885, 1464, 1807]])]

    second_batch = next(data_iter)
    print(second_batch) # [tensor([[ 367, 2885, 1464, 1807]]), tensor([[2885, 1464, 1807, 3619]])]

    inputs, targets = next(data_iter)
    print(f"input:{inputs,},targets:{targets}")