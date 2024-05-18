import torch
from prettytable import PrettyTable

def reduce_size(embeddings, n_size = 1, m_size = 8):
    second_dim = list(embeddings.shape)[-1]
    first_dim = list(embeddings.shape)[0]
    embeddings = torch.reshape(embeddings, (first_dim, int(second_dim/(n_size*m_size)), n_size*m_size))
    embeddings = torch.sum(embeddings, (1), keepdim = True).squeeze()
    return embeddings

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