import torch
from torch import nn
from transformers import AutoModel

from models import EfficientKAN, FastKAN, FasterKAN 

class TransformerEnsembleKAN(nn.Module):
    def __init__(
        self,
        hiddens_layer,
        model_name
        
    ) -> None:
        super().__init__()
        #self.drop = torch.nn.Dropout(p=0.1) 
        self.model = AutoModel.from_pretrained(model_name)
        
        self.efficient_kan = FastKAN(hiddens_layer) 
        self.fast_kan = FastKAN(hiddens_layer) 
        self.faster_kan = FasterKAN(hiddens_layer) 

    def forward(self, input_ids, attention_mask):
        _, x = self.model(input_ids=input_ids, attention_mask=attention_mask, return_dict=False)
        x1 = self.efficient_kan(x)
        x2 = self.fast_kan(x)
        x3 = self.faster_kan(x)
        x = x1 + x2 + x3 
        return x
        

       