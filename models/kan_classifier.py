from torch import nn
from kan import KAN
import torch

class KANClassifier(nn.Module):
    def __init__(self, n_input, n_hidden, n_class, grid, k):
        super(KANClassifier, self).__init__()
        self.KAN = KAN(width = [n_input, n_hidden, n_class], grid = grid, k = k)

    def forward(self, x):
        output = self.KAN(x)
        output = nn.Sigmoid()(output)
        output = torch.squeeze(output)
        return output