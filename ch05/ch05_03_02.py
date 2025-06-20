import torch
import matplotlib.pyplot as plt

vocab = {
    "closer": 0,
    "every": 1,
    "effort": 2,
    "forward": 3,
    "inches": 4,
    "moves": 5,
    "pizza": 6,
    "toward": 7,
    "you": 8
}

inverse_vocab = {v: k for k, v in vocab.items()}
print(inverse_vocab)

next_token_logits = torch.tensor([4.51, 0.89, -1.90, 6.75, 1.63, -1.62, -1.89, 6.28, 1.79])

probas = torch.softmax(next_token_logits, dim=0)
torch.set_printoptions(sci_mode=False)  # 设置打印选项，禁用科学计数法
"""
输出概率分布：
Probas: tensor([    0.0609,     0.0016,     0.0001,     0.5721,     0.0034,     0.0001,
            0.0001,     0.3576,     0.0040])
"""
print("Probas:", probas)

top_k = 3
top_logits, top_pos = torch.topk(next_token_logits, top_k)
print("top logits:", top_logits)    # top logits: tensor([6.7500, 6.2800, 4.5100])
print("top pos:", top_pos)  # top pos: tensor([3, 7, 0])

# 将识别比前3个logits值还要低的logits值置为负无穷
new_logits = torch.where(condition=next_token_logits < top_logits[-1], input=torch.tensor(float('-inf')), other=next_token_logits)
print("new logits:", new_logits)    # new logits: tensor([4.5100,   -inf,   -inf, 6.7500,   -inf,   -inf,   -inf, 6.2800,   -inf])

# 包含3个非零概率分数的向量
topk_probas = torch.softmax(new_logits, dim=0)
print("topk probas:", topk_probas)  # topk probas: tensor([0.0615, 0.0000, 0.0000, 0.5775, 0.0000, 0.0000, 0.0000, 0.3610, 0.0000])