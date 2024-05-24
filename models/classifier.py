from torch import nn
from transformers import AutoModel

class TransformerClassifier(nn.Module):
    def __init__(self, n_classes, model_name = 'bert-base-cased'):
        super(TransformerClassifier, self).__init__()
        self.model = AutoModel.from_pretrained(model_name)
        self.drop = nn.Dropout(p=0.1) # dropout
        self.out = nn.Linear(self.model.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask, return_dict=False):
        _, pooled_output = self.model(input_ids=input_ids, attention_mask=attention_mask, return_dict=return_dict)
        output = self.drop(pooled_output) # dropout
        return self.out(output)
       