import torch

torch.manual_seed(123)
log_probas = torch.log(torch.tensor([4.3559e-01, 2.2607e-01, 1.1023e-01, 1.3012e-01, 6.0395e-01, 4.8063e-01]))
print(log_probas)

avg_log_probas = torch.mean(log_probas)
print(avg_log_probas)

neg_avg_log_probas = avg_log_probas * -1
print(neg_avg_log_probas)

perplexity = torch.exp(neg_avg_log_probas)
print(perplexity)