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

    model = GPTModel(BASE_CONFIG)
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

    r"""
        GPTModel(
    (tok_emb): Embedding(50257, 768)
    (pos_emb): Embedding(1024, 768)
    (drop_emb): Dropout(p=0.0, inplace=False)
    (trf_blocks): Sequential(
        ......
        (11): TransformerBlock(
        (att): MultiHeadAttention(
            (W_query): Linear(in_features=768, out_features=768, bias=True)
            (W_key): Linear(in_features=768, out_features=768, bias=True)
            (W_value): Linear(in_features=768, out_features=768, bias=True)
            (out_proj): Linear(in_features=768, out_features=768, bias=True)
            (dropout): Dropout(p=0.0, inplace=False)
        )
        (ff): FeedForward(
            (layers): Sequential(
            (0): Linear(in_features=768, out_features=3072, bias=True)
            (1): GELU()
            (2): Linear(in_features=3072, out_features=768, bias=True)
            )
        )
        (norm1): LayerNorm()
        (norm2): LayerNorm()
        (drop_shortcut): Dropout(p=0.0, inplace=False)
        )
    )
    (final_norm): LayerNorm()
    (out_head): Linear(in_features=768, out_features=50257, bias=False)
    )
    """
    print(model)