from datasets import load_dataset
from efficient_kan import KAN
from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import CountVectorizer
from transformers import BertModel, BertTokenizer
from torch.utils.data import DataLoader
from tqdm import tqdm

import numpy as np
import pandas as pd
import sys
import torch
import torch.nn as nn
import torch.optim as optim

import gc

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device_cpu = torch.device("cpu")

def create_data_loader(dataset, tokenizer, max_len = 512, batch_size = 4, shuffle = False):

    texts, labels = [], []
 
    for item in dataset:
        text = '[CLS] ' + item['sentence1'] + ' [SEP] ' + item['sentence2']
        texts.append(text)
        labels.append(item['label'])
        
    ds = PreparedDataset(texts= np.array(texts),
                         labels = np.array(labels),
                         tokenizer=tokenizer,
                         max_len=max_len)
                         
    return DataLoader(ds, batch_size=batch_size, num_workers=4, shuffle = shuffle)
    
class PreparedDataset():
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = str(self.texts[item])
        label = self.labels[item]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            #pad_to_max_length=True,
            padding = "max_length",
            return_attention_mask=True,
            truncation=True,
            return_tensors='pt',
            )
        return { 
                'text': text, 
                'input_ids': encoding['input_ids'].flatten(), 
                'attention_mask': encoding['attention_mask'].flatten(),
                'label': torch.tensor(label, dtype=torch.long)
               }

def get_embeddings(data, n_size = 1, m_size = 768, embed_type = 'po'):
    
    with torch.no_grad():
        input_ids = data["input_ids"].to(device)
        attention_mask = data["attention_mask"].to(device)
        
        outputs = trans_model(input_ids=input_ids, attention_mask=attention_mask) #return_dict = True    
        #embedding_matrix = model.embeddings.word_embeddings.weight
        #embeddings = embedding_matrix[input_ids] # 32, 512, 768
        
        # pooled_outputs 
        #return outputs['last_hidden_state'] 32, 512, 768
        #return outputs[-1] # 32, 768
    
        embeddings = {}
        if (embed_type == 'lhs'): #last_hidden_state
            embeddings = outputs['last_hidden_state']
            embeddings = torch.sum(hiddens, (1), keepdim = True) # consume much memory
        elif (embed_type == 'po'): #pooled_outputs 
            embeddings = outputs[-1]
        
        del outputs
        torch.cuda.empty_cache()
        gc.collect()
        return embeddings
    
def train_kan(trainloader, valloader, epochs = 20, n_size = 1, m_size = 768, n_hidden = 64, n_class = 2):
    
    # define KAN model
    model = KAN([n_size*m_size, n_hidden, n_class])
    model.to(device_cpu)
    # define optimizer
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    # define learning rate scheduler
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8)
    # define loss
    criterion = nn.CrossEntropyLoss()
    best_accuracy = 0
    
    for epoch in range(epochs):
        # Train
        model.train()
        trans_model.eval() 
        
        with tqdm(trainloader) as pbar:
            for i, items in enumerate(pbar):
                #texts = get_embeddings(items).view(-1, n_size*m_size).to(device)
                texts = get_embeddings(items).to(device_cpu)
                labels = items['label']
                optimizer.zero_grad()
                output = model(texts)
                loss = criterion(output, labels.to(device_cpu))
                loss.backward()
                optimizer.step()
                accuracy = (output.argmax(dim=1) == labels.to(device_cpu)).float().mean()
                #elapsed = pbar.format_dict['elapsed']
                #elapsed_str = pbar.format_interval(elapsed)
                
                pbar.set_postfix(loss=loss.item(), accuracy=accuracy.item(), lr=optimizer.param_groups[0]['lr'])
                

        # validation
        model.eval()
        val_loss = 0
        val_accuracy = 0
        
        with torch.no_grad():
            with tqdm(valloader) as pbar:
                for i, items in enumerate(pbar):
                    #texts = get_embeddings(items).view(-1, n_size*m_size).to(device)
                    texts = get_embeddings(items).to(device_cpu)
                    labels = items['label']
                    output = model(texts)
                    val_loss += criterion(output, labels.to(device_cpu)).item()
                    #preds = torch.argmax(output, dim=1)
                    val_accuracy += ((output.argmax(dim=1) == labels.to(device_cpu)).float().mean().item())
                    
                    pbar.set_postfix(val_loss=val_loss/len(valloader), val_accuracy=val_accuracy/len(valloader))
                    del output
        val_loss /= len(valloader)
        val_accuracy /= len(valloader)
        
        if (val_accuracy > best_accuracy):
            best_accuracy = val_accuracy
            torch.save(model, 'model.pth')

        # update learning rate
        scheduler.step()

        print(
            f"Epoch {epoch + 1}, Val Loss: {val_loss}, Val Accuracy: {val_accuracy}"
        )
        torch.cuda.empty_cache()
        gc.collect()


def infer_kan(testloader, model_name = 'model.pth', n_size = 24, m_size = 32):

    
    model = torch.load(model_name)
    model.eval()

    criterion = nn.CrossEntropyLoss()
    test_loss = 0
    test_accuracy = 0
    with torch.no_grad():
        with tqdm(testloader) as pbar:
            for i, items in enumerate(pbar):
                texts = get_embeddings(items).view(-1, n_size*m_size).to(device)
                labels = items['label']
                output = model(texts)
                test_loss += criterion(output, labels.to(device)).item()
                test_accuracy += ((output.argmax(dim=1) == labels.to(device)).float().mean().item())
                pbar.set_postfix(test_loss=test_loss.item(), test_accuracy=test_accuracy.item())

        test_loss /= len(testloader)
        test_accuracy /= len(testloader)

    print(test_accuracy, test_loss)
    
if __name__ == "__main__":
    
    dataset = load_dataset('glue', 'mrpc')
    train_set = dataset['train']
    print('train example: ', train_set[0])
    val_set = dataset['validation']
    print('val example: ', val_set[0])
    test_set = dataset['test']
    print('data size: ', len(train_set), len(val_set), len(test_set)) # 3668 408 1725
    
    #google-bert/bert-large-cased
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
    
    global trans_model
    trans_model = BertModel.from_pretrained('bert-base-cased')
    trans_model = trans_model.to(device)

    trainloader = create_data_loader(train_set, tokenizer, max_len = 512, batch_size = 2, shuffle = True)
    valloader = create_data_loader(val_set, tokenizer, max_len = 512, batch_size = 2)
    print('Load data done!')
    
    train_kan(trainloader, valloader, epochs = 5)
    

'''final_test = []
for x, y in zip(x_test, y_test): final_test.append( (x.toarray().astype(np.float32).reshape(1, n_features, n_features), int(y)) )'''

'''
# load data
trainloader = DataLoader(final_train, batch_size=64, shuffle=True)
valloader = DataLoader(final_val, batch_size=64, shuffle=False)
#testloader = DataLoader(final_test, batch_size=64, shuffle=False)

i = 0 
for item in trainloader:
    print(item[0].size())
    if (i == 0): break
'''

'''
text = tokenizer.encode_plus(
                all_data[0],
                add_special_tokens=True,
                max_length=512,
                return_token_type_ids=False,
                #pad_to_max_length=True,
                padding = "max_length",
                return_attention_mask=True,
                truncation=True,
                return_tensors='pt',
                )

print('text: ', text)

input_ids = text["input_ids"].to(device)
attention_mask = text["attention_mask"].to(device)
outputs = model(input_ids=input_ids, attention_mask=attention_mask)
print('outputs: ', outputs['last_hidden_state'], len(outputs['last_hidden_state']))'''
