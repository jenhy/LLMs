import torch
import tiktoken
import sys, os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from ch05.gpt_download  import download_and_load_gpt2
from ch04.ch04_04_06  import GPTModel
from ch05.ch05_05_01  import load_weights_into_gpt
from ch06.ch06_06_03 import SpamDataset
from torch.utils.data import DataLoader
from ch05.ch05_01_03 import calc_loss_batch, calc_loss_loader, get_data_loaders


def calc_accuracy_loader(data_loader, model, device, num_batches=None):
    model.eval()
    correct_predictions, num_examples = 0, 0

    if num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))

    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            input_batch = input_batch.to(device)
            target_batch = target_batch.to(device)

            with torch.no_grad():
                logits = model(input_batch)[:, -1, :]
            predicted_labels = torch.argmax(logits, dim=-1)

            num_examples += predicted_labels.shape[0]
            correct_predictions += ((predicted_labels == target_batch).sum().item())
        else:
            break
    
    return correct_predictions / num_examples


if  __name__ == "__main__":    
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
    model.to(device)
    torch.manual_seed(123)
    load_weights_into_gpt(model, params)
    model.eval()

    # 在分类微调之前，先将模型参数冻结，即所有层的参数都不可以更新
    for param in model.parameters():
        param.requires_grad = False

    torch.manual_seed(123)

    # 替换输出层(必须的)，原来是50257维，词汇表大小，现在改为2维
    num_classes = 2
    model.out_head = torch.nn.Linear(in_features=BASE_CONFIG["emb_dim"], out_features=num_classes)

    model.to(device)
    print(next(model.parameters()).device)  # 应该输出 cuda:0
    
    for param in model.trf_blocks[-1].parameters():
        param.requires_grad = True
    for param in model.final_norm.parameters():
        param.requires_grad = True

    train_dataset = SpamDataset(csv_file="train.csv", max_length=None, tokenizer=tokenizer)
    val_dataset = SpamDataset(csv_file="validation.csv", max_length=train_dataset.max_length, tokenizer=tokenizer)
    test_dataset = SpamDataset(csv_file="test.csv", max_length=train_dataset.max_length, tokenizer=tokenizer)

    train_loader = DataLoader(dataset=train_dataset, batch_size=10, shuffle=True, num_workers=0, drop_last=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=10, shuffle=False, num_workers=0, drop_last=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=10, shuffle=False, num_workers=0, drop_last=False)

    train_accuracy = calc_accuracy_loader(train_loader, model, device, num_batches=10)
    val_accuracy = calc_accuracy_loader(val_loader, model, device, num_batches=10)
    test_accuracy = calc_accuracy_loader(test_loader, model, device, num_batches=10)

    print(f"Train accuracy: {train_accuracy*100:.2f}%")
    print(f"Validation accuracy: {val_accuracy*100:.2f}%")
    print(f"Test accuracy: {test_accuracy*100:.2f}%")

    with torch.no_grad():
        train_losss = calc_loss_loader(train_loader, model, device, num_batches=10)
        val_losss = calc_loss_loader(val_loader, model, device, num_batches=10)
        test_losss = calc_loss_loader(test_loader, model, device, num_batches=10)

    """
    Training loss: 2.422
    Validation loss: 2.579
    Test loss: 2.518
    """
    print(f"Training loss: {train_losss:.3f}")
    print(f"Validation loss: {val_losss:.3f}")
    print(f"Test loss: {test_losss:.3f}")
