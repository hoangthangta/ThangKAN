import torch
from prettytable import PrettyTable

def reduce_size(embeddings, n_size = 1, m_size = 8):
    
    em_len = len(embeddings) # length of embeddings
    embeddings = torch.Tensor(embeddings)
    embeddings = torch.reshape(embeddings, (n_size, int(em_len/m_size), m_size))
    embeddings = torch.sum(embeddings, (1), keepdim = True).squeeze()
    return embeddings.tolist()

def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params += params
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params