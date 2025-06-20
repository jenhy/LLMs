import torch
import tiktoken
import sys, os
import time
import matplotlib.pyplot as plt
from IPython.display import display

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from ch05.gpt_download  import download_and_load_gpt2
from ch04.ch04_04_06  import GPTModel
from ch05.ch05_01_03 import calc_loss_batch
from ch05.ch05_02_01 import evaluate_model
from ch06.ch06_06_06 import calc_accuracy_loader
from ch06.ch06_06_03 import SpamDataset
from torch.utils.data import DataLoader

def train_classifier_simple(model, train_loader, val_loader, optimizer, device, num_epochs, eval_freq, eval_iter):
    """
    微调模型进行分类的主函数。
    参数解释：
    model: 这里是GPT-2模型。
    train_loader: 训练数据加载器。
    val_loader: 验证数据加载器。
    optimizer: 优化器。
    device: 设备（如 'cpu' 或 'cuda'），控制数据/模型的计算在哪进行。
    num_epochs: 训练的轮数。
    eval_freq: 模型评估的频率。也就是多少步打印一下
    eval_iter:  模型评估的迭代次数。
    返回：
    跟踪训练数据损失、验证数据损失和分类。
    """

    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    examples_seen, global_step = 0, -1

    for epoch in range(num_epochs):
        model.train()
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()
            optimizer.step()
            examples_seen += input_batch.shape[0]
            global_step += 1

            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                print(f"Ep {epoch+1} (Step {global_step:06d}): Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")

        train_accuracy = calc_accuracy_loader(train_loader, model, device, num_batches=eval_iter)
        val_accuracy = calc_accuracy_loader(val_loader, model, device, num_batches=eval_iter)

        train_accs.append(train_accuracy)
        val_accs.append(val_accuracy)

    return train_losses, val_losses, train_accs, val_accs, examples_seen

def plot_values(epochs_seen, examples_seen, train_values, val_values, label="loss"):
    fig, ax1 = plt.subplots(figsize=(5, 3))
    ax1.plot(epochs_seen, train_values, label=f"Training {label}")
    ax1.plot(epochs_seen, val_values, linestyle="-.", label=f"Validation {label}")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel(label)
    ax1.legend()

    ax2 = ax1.twiny()
    ax2.plot(examples_seen, train_values, alpha=0)
    ax2.set_xlabel("Examples seen")

    fig.tight_layout()
    display(fig)
    plt.savefig(f"{label}-plot.pdf")
    # plt.show()

if __name__ == "__main__":
    start_time = time.time()
    torch.manual_seed(123)

    tokenizer = tiktoken.get_encoding("gpt2")
    CHOOSE_MODEL = "gpt2-small (124M)"

    BASE_CONFIG = {
        "vocab_size": 50257,
        "context_length": 1024,
        "drop_rate": 0.0,
        "qkv_bias": True
    }

    model_configs = {
        "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
        "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
        "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
        "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25}    
    }

    BASE_CONFIG.update(model_configs[CHOOSE_MODEL])

    model_size = CHOOSE_MODEL.split(" ")[-1].lstrip("(").rstrip(")")
    settings, params = download_and_load_gpt2(model_size=model_size, models_dir="gpt2")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GPTModel(BASE_CONFIG)

    # 在分类微调之前，先将模型参数冻结，即所有层的参数都不可以更新
    for param in model.parameters():
        param.requires_grad = False

    # 替换输出层(必须的)，原来是50257维，词汇表大小，现在改为2维
    # 这里我犯了一个错误，没有替换输出层的维度，导致输出还是50257维
    num_classes = 2
    model.out_head = torch.nn.Linear(in_features=BASE_CONFIG["emb_dim"], out_features=num_classes)

    # 这里犯了一个错误，替换输出层默认在CPU上执行，导致失败，所以需要再执行一次
    # 或者上面一行代码改成torch.nn.Linear(in_features=BASE_CONFIG["emb_dim"], out_features=num_classes).to(device)
    model.to(device)
    print(next(model.parameters()).device)  # 应该输出 cuda:0

    # 解冻最后一个块和最终层
    for param in model.trf_blocks[-1].parameters():
        param.requires_grad = True
    for param in model.final_norm.parameters():
        param.requires_grad = True

    # 再初始化 optimizer（此时只有 requires_grad=True 的参数会被优化）
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.1)

    train_dataset = SpamDataset(csv_file="train.csv", max_length=None, tokenizer=tokenizer)
    val_dataset = SpamDataset(csv_file="validation.csv", max_length=train_dataset.max_length, tokenizer=tokenizer)
    test_dataset = SpamDataset(csv_file="test.csv", max_length=train_dataset.max_length, tokenizer=tokenizer)

    batch_size = 5
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=False)

    num_epochs = 5
    train_losses, val_losses, train_accs, val_accs, examples_seen = train_classifier_simple(model, train_loader, val_loader, optimizer, device, num_epochs=num_epochs, eval_freq=50, eval_iter=5)

    end_time = time.time()
    execution_time_minutes = (end_time - start_time) / 60
    print(f"Training completed in {execution_time_minutes:.2f} minutes.")

    # 绘制损失
    epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
    examples_seen_tensor = torch.linspace(0, examples_seen, len(train_losses))
    plot_values(epochs_tensor, examples_seen_tensor, train_losses, val_losses)

    # 绘制准确率
    epochs_tensor = torch.linspace(0, num_epochs, len(train_accs))
    examples_seen_tensor = torch.linspace(0, examples_seen, len(train_accs))
    plot_values(epochs_tensor, examples_seen_tensor, train_accs, val_accs, label="accuracy")

    # 评估精度，eval_iter为None
    train_accuracy = calc_accuracy_loader(train_loader, model, device)
    val_accuracy = calc_accuracy_loader(val_loader, model, device)
    test_accuracy = calc_accuracy_loader(test_loader, model, device)

    print(f"Train accuracy: {train_accuracy*100:.2f}%")
    print(f"Val accuracy: {val_accuracy*100:.2f}%")
    print(f"Test accuracy: {test_accuracy*100:.2f}%")