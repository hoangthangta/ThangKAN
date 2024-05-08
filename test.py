import numpy as np
from scipy.sparse import csr_matrix
import torch

__author__ = 'Andrea Esuli'

Acsr = csr_matrix([[1, 2, 0], [0, 0, 3], [4, 0, 5]])
print('Acsr',Acsr)

Acoo = Acsr.tocoo()
print('Acoo',Acoo)

Apt = torch.sparse.LongTensor(torch.LongTensor([Acoo.row.tolist(), Acoo.col.tolist()]),
                              torch.LongTensor(Acoo.data.astype(np.int32)))
print('Apt',Apt)