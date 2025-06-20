import torch
from torch.utils.data import Dataset, DataLoader

# 自定义Dataset类
class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

# 创建数据集
data = [1, 2, 3, 4, 5]
dataset = MyDataset(data)

# 直接调用 __len__ 和 __getitem__ 方法
print("直接调用 __len__ 方法:", dataset.__len__())
print("直接调用 __getitem__ 方法:", dataset.__getitem__(2))

# 使用 Python 内置函数和操作符调用
print("使用 len() 函数:", len(dataset))
print("使用索引操作符:", dataset[2])

# 与 DataLoader 结合使用
# DataLoader：这是torch.utils.data模块里的一个类，其作用是对Dataset对象的数据进行批量加载，并且支持多线程加载、打乱数据等功能。
# dataset：这里传入的是之前自定义的MyDataset类的实例dataset。这个实例包含了我们想要处理的数据。
# batch_size=2：此参数指定了每个批次包含的样本数量。在这个例子中，每个批次会有 2 个样本。
# shuffle=True：该参数表明在每个训练周期开始时，是否对数据进行打乱。设置为True意味着每次迭代数据时，数据的顺序都会被打乱，这样有助于提高模型的泛化能力。
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
for batch in dataloader:
    print("DataLoader 返回的批次:", batch)
    