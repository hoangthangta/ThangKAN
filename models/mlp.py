import torch
import torch.nn as nn
from transformers import AutoModel

class TransformerMLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, model_name):
        super(TransformerMLP, self).__init__()
        self.model = AutoModel.from_pretrained(model_name) # BERT
        self.fc_layers = nn.ModuleList([nn.Linear(input_size, hidden_sizes[0])])
        for i in range(len(hidden_sizes) - 1):
            self.fc_layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
        self.output_layer = nn.Linear(hidden_sizes[-1], output_size)

    def forward(self, input_ids, attention_mask):
        output = self.model(input_ids=input_ids, attention_mask=attention_mask)[0]
        output = output[:, 0, :]  # Using [CLS] token representation
        for fc_layer in self.fc_layers:
            output = torch.relu(fc_layer(output))
        output = self.output_layer(output)
        return output
