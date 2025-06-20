import tiktoken # 导入 tiktoken 库，用于处理文本的编码和分词。tiktoken 库提供了许多预训练的编码器，可以针对不同的语言和模型进行编码和分词。
import torch
import pandas as pd # 导入数据处理库，这里用于读取 CSV 文件并操作其中的数据。
from torch.utils.data import Dataset    # 导入 PyTorch 的数据处理库。自定义数据集类必须继承自 Dataset 才能与 DataLoader 一起配合使用，实现数据的迭代读取。
from torch.utils.data import DataLoader

class SpamDataset(Dataset):
    """
    定义一个继承自 Dataset 的自定义类，用于文本与标签的加载和预处理。
    让这个类可以与 PyTorch 的 DataLoader 协同使用，实现按需读取样本。
    """
    def __init__(self, csv_file, tokenizer, max_length=None, pad_token_id=50256):
        """
        初始化数据集对象，接收 CSV 文件路径、tokenizer、最大长度进行裁剪和填充、padding 的 token ID。
        """

        # 使用 pandas 读取 CSV 文件并保存为 DataFrame
        self.data = pd.read_csv(csv_file)

        # 使用列表推导式，把所有文本列中的字符串用 tokenizer 编码成整数列表（tokens）。
        self.encoded_texts = [tokenizer.encode(text) for text in self.data["Text"]]

        # 最大长度处理
        # 控制是否使用固定的最大长度；如果未提供，则自动找出最大长度。
        # 若 max_length=None，调用辅助函数 _longest_encoded_length。
        # 若指定了 max_length，则对所有样本进行截断。
        if max_length is None:
            self.max_length = self._longest_encoded_length()
        else:
            self.max_length = max_length
            self.encoded_texts = [encoded_text[:self.max_length] for encoded_text in self.encoded_texts]

        # 填充所有文本为相同的长度。将不足 max_length 的 token 序列进行填充（padding）
        # 对每个 token 列表，在末尾添加 pad_token_id 若干次，使其达到最大长度。
        self.encoded_texts = [encoded_text + [pad_token_id] * (self.max_length - len(encoded_text)) for encoded_text in self.encoded_texts]

    def __getitem__(self, index):
        """
        定义索引访问函数，支持 dataset[i] 的调用。按下标取出第 index 个样本的 token 和 label，并转为 PyTorch 的 tensor。
        """

        # 获取第 index 个文本的编码结果。
        encoded = self.encoded_texts[index]

        # 获取第 index 行的标签值。使用 iloc 按位置索引 DataFrame 中的行，并取 Label 列。
        label = self.data.iloc[index]["Label"]

        # 返回编码后的文本和对应的标签的张量
        return (torch.tensor(encoded, dtype=torch.long), torch.tensor(label, dtype=torch.long))

    def __len__(self):
        """
        返回数据集中的样本数量。
        """
        return len(self.data)
    
    def _longest_encoded_length(self):
        """
        一个内部辅助函数，用于计算文本中最长的一条编码长度。遍历所有 encoded_text，记录最大长度并返回。
        """
        max_length = 0
        for encoded_text in self.encoded_texts:
            encoded_length = len(encoded_text)
            if encoded_length > max_length:
                max_length = encoded_length
    
        return max_length
    

    
if  __name__ == "__main__":
    tokenizer = tiktoken.get_encoding("gpt2")   #  初始化一个 GPT-2 专用的 tokenizer

    # 打印特殊 token <|endoftext|> 的 token 编码[50256]，通过 allowed_special 显式允许特殊符号
    print(tokenizer.encode("<|endoftext|>", allowed_special={"<|endoftext|>"}))

    # 创建训练集数据对象
    train_dataset = SpamDataset(csv_file="train.csv", max_length=None, tokenizer=tokenizer)
    # 获取最长的编码长度120
    print(train_dataset.max_length)

    # 创建验证集和测试集数据对象。明确传入和训练集相同的 max_length，以确保三者的输入维度一致。
    val_dataset = SpamDataset(csv_file="validation.csv", max_length=train_dataset.max_length, tokenizer=tokenizer)
    # 获取最长的编码长度120
    print(val_dataset.max_length)

    test_dataset = SpamDataset(csv_file="test.csv", max_length=train_dataset.max_length, tokenizer=tokenizer)
    # 获取最长的编码长度120
    print(test_dataset.max_length)

    num_workers = 0
    batch_size = 8
    torch.manual_seed(123)

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False)

    for input_batch, target_batch in train_loader:
        pass

    # torch.Size([8, 120])
    print(input_batch.shape)
    print(input_batch.shape[0])
    print(input_batch.shape[1])

    # torch.Size([8])
    print(target_batch.shape)

    """
    130 training batches in total
    19 validation batches in total
    38 test batches in total
    """
    print(f"{len(train_loader)} training batches in total")
    print(f"{len(val_loader)} validation batches in total")
    print(f"{len(test_loader)} test batches in total")

    # 打印第一批数据
    data_iter = iter(train_loader)
    first_batch = next(data_iter)
    print(first_batch)

