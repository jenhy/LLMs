import torch
import tiktoken
import sys, os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from ch05.gpt_download  import download_and_load_gpt2
from ch04.ch04_04_06  import GPTModel
from ch05.ch05_05_01  import load_weights_into_gpt
from ch04.ch04_04_07 import generate_text_simple
from ch05.ch05_01_01 import text_to_token_ids, token_ids_to_text

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
    load_weights_into_gpt(model, params)
    model.eval()

    text_1 = "Every effort moves you"
    token_ids = generate_text_simple(model=model, idx=text_to_token_ids(text=text_1, tokenizer=tokenizer), max_new_tokens=15,
                         context_size=BASE_CONFIG["context_length"])
    print(token_ids_to_text(token_ids=token_ids, tokenizer=tokenizer))

    text_2 =(
        "Is the following text 'spam'? Answer with 'yes' or 'no':"
        " 'You are a winner you have been specially"
        " selected to receive $1000 cash or a $2000 award.'"
    )
    token_ids = generate_text_simple(model=model, idx=text_to_token_ids(text=text_2, tokenizer=tokenizer), max_new_tokens=23,
                         context_size=BASE_CONFIG["context_length"])
    print(token_ids_to_text(token_ids=token_ids, tokenizer=tokenizer))

    # print(model)

    # 在分类微调之前，先将模型参数冻结，即所有层的参数都不可以更新
    for param in model.parameters():
        param.requires_grad = False

    torch.manual_seed(123)
    num_classes = 2
    model.out_head = torch.nn.Linear(in_features=BASE_CONFIG["emb_dim"], out_features=num_classes)

    model.to(device)
    print(next(model.parameters()).device)  # 应该输出 cuda:0

    for param in model.trf_blocks[-1].parameters():
        param.requires_grad = True
    for param in model.final_norm.parameters():
        param.requires_grad = True

    inputs = tokenizer.encode("Do you have time")
    inputs = torch.tensor(inputs).unsqueeze(0)

    with torch.no_grad():
        outputs = model(inputs)
    
    """Outputs:
    tensor([[[-1.5854,  0.9904],
            [-3.7235,  7.4548],
            [-2.2661,  6.6049],
            [-3.5983,  3.9902]]])"""
    print("Outputs:\n", outputs)

    """
    Outputs shape:
    torch.Size([1, 4, 2])  
    """
    print("Outputs shape:\n", outputs.shape)
    print("Last Outputs token:", outputs[:, -1, :])
    print("First Outputs token:", outputs[:, 0, :])

    probas = torch.softmax(outputs[:, -1, :], dim=-1)
    print("Probas:\n", probas)
    label = torch.argmax(probas)
    print("Class label:", label.item())

    logits = outputs[:, -1, :]
    label = torch.argmax(logits)
    print("Class label:", label.item())