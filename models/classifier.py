from torch import nn
from transformers import AutoModel

class CategoryClassifier(nn.Module):
    def __init__(self, n_classes, pretrained_model = 'bert-base-cased'):
        super(CategoryClassifier, self).__init__()
        self.model = AutoModel.from_pretrained(pretrained_model)
        self.drop = nn.Dropout(p=0.1)
        self.out = nn.Linear(self.model.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask, return_dict=False):
        _, pooled_output = self.model(input_ids=input_ids, attention_mask=attention_mask, return_dict=return_dict)
        output = self.drop(pooled_output)
        return self.out(output)