from kan import KAN
from torch import nn
import torch

class WrapKAN(nn.Module):

    def __init__(self, n_input, n_hidden, n_class, grid, k, device):
        super(WrapKAN, self).__init__()
        self.KAN = KAN(width = [n_input, n_hidden, n_class], grid = grid, k = k, device = device)
        
    def forward(self, x):
        output = self.KAN(x)
        output = nn.Sigmoid()(output)
        output = torch.squeeze(output)
        return output
