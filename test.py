import torch.nn as nn
import torch


x, y = torch.rand(64, 100), torch.rand(64, 100)

print(x, x.shape)
print(y, y.shape)

z = torch.cat((x, y), -1)

print(z, z.shape)
