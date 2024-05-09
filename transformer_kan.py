from datasets import load_dataset
from efficient_kan import KAN
from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import CountVectorizer
from transformers import AutoModel, AutoTokenizer
from torch.utils.data import DataLoader
from tqdm import tqdm

import argparse
import numpy as np
import pandas as pd
import sys
import torch
import torch.nn as nn
import torch.optim as optim

import gc

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device_cpu = torch.device("cpu")

from pathlib import Path
Path("output").mkdir(parents=True, exist_ok=True)

def create_data_loader(ds_name, dataset, tokenizer, max_len = 512, batch_size = 4, shuffle = False):

    texts, labels = [], []
 
    for item in dataset:
        text = '[CLS] ' + item['sentence1'] + ' [SEP] ' + item['sentence2']
        texts.append(text)
        labels.append(item['label'])
        
    ds = TextDataset(texts= np.array(texts),
                         labels = np.array(labels),
                         tokenizer=tokenizer,
                         max_len=max_len)
                         
    return DataLoader(ds, batch_size=batch_size, num_workers=4, shuffle = shuffle)
    
class TextDataset():
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
        
        outputs = em_model(input_ids=input_ids, attention_mask=attention_mask) #return_dict = True    
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

    
def train_kan(trainloader, valloader, ds_name = 'mrpc', em_model_name = 'bert-base-cased', epochs = 20, n_size = 1, m_size = 768, n_hidden = 64, n_class = 2):
    
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
    
    model_id = ''
    try:
        model_id = em_model_name.split('/')[1]
    except:
        model_id = em_model_name
        
    saved_model_name =  model_id + '_' +  ds_name + '_model.pth'
    
    for epoch in range(epochs):
        # Train
        model.train()
        em_model.eval() 
        
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
            torch.save(model, 'output/saved_model_name')

        # update learning rate
        scheduler.step()

        print(
            f"Epoch {epoch + 1}, Val Loss: {val_loss}, Val Accuracy: {val_accuracy}"
        )
        torch.cuda.empty_cache()
        gc.collect()


def infer_kan(test_loader, model_path = 'model.pth', n_size = 24, m_size = 32):

    model = torch.load(model_path)
    model.eval()

    criterion = nn.CrossEntropyLoss()
    test_loss = 0
    test_accuracy = 0
    with torch.no_grad():
        with tqdm(test_loader) as pbar:
            for i, items in enumerate(pbar):
                texts = get_embeddings(items).view(-1, n_size*m_size).to(device)
                labels = items['label']
                output = model(texts)
                test_loss += criterion(output, labels.to(device)).item()
                test_accuracy += ((output.argmax(dim=1) == labels.to(device)).float().mean().item())
                pbar.set_postfix(test_loss=test_loss.item(), test_accuracy=test_accuracy.item())

        test_loss /= len(test_loader)
        test_accuracy /= len(test_loader)

    print(test_accuracy, test_loss)
    return {'accuracy':test_accuracy, 'avg_loss':test_loss}

def prepare_dataset(ds_name = 'mrpc'):
    
    dataset = []
    dataset = load_dataset('glue', ds_name)
    return dataset

def build_data_loader(ds_name, em_model_name,  max_len = 512, batch_size = 4, test_only = False):
    
    dataset = prepare_dataset(ds_name)
    tokenizer = AutoTokenizer.from_pretrained(em_model_name)
    
    global em_model
    em_model = AutoModel.from_pretrained(em_model_name)
    em_model = em_model.to(device)
    
    train_loader, val_loader, test_loader = [], [], []
    test_loader = create_data_loader(ds_name, dataset['test'], tokenizer, max_len = max_len, batch_size = batch_size)
    if (test_only == False):
        train_loader = create_data_loader(ds_name, dataset['train'], tokenizer, max_len = max_len, \
                                            batch_size = batch_size, shuffle = True)
        val_loader = create_data_loader(ds_name, dataset['validation'], tokenizer, \
                                            max_len = max_len, batch_size = batch_size)
    

    return {'train': train_loader, 'validation': val_loader, 'test': test_loader}

def main(args):
    if (args.mode == 'train'):
        loader = build_data_loader(ds_name = args.ds_name, em_model_name = args.em_model_name, max_len = args.max_len, batch_size = args.batch_size)
        #train_kan(loader['train'], loader['validation'], ds_name = args.ds_name, em_model_name = args.em_model_name, epochs = args.epochs, \
        #                n_size = args.n_size, m_size = args.m_size, n_hidden = args.n_hidden, n_class = args.n_class)

    elif (args.mode == 'test'):
        loader = build_data_loader(ds_name = args.ds_name, max_len = args.max_len, batch_size = args.batch_size, test_only = True)
        infer_kan(loader['test'], model_path = args.model_path, n_size = args.n_size, m_size = args.m_size)
        
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Training Parameters')
    parser.add_argument('--mode', type=str, default='train') # or test
    parser.add_argument('--em_model_name', type=str, default='bert-base-cased') # or test
    parser.add_argument('--ds_name', type=str, default='mrpc')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--max_len', type=int, default=512)
    
    parser.add_argument('--n_size', type=int, default=1)
    parser.add_argument('--m_size', type=int, default=768)
    parser.add_argument('--n_hidden', type=int, default=64)
    parser.add_argument('--n_class', type=int, default=2)
    
    '''parser.add_argument('--train_path', type=str, default='dataset/train.json') 
    parser.add_argument('--test_path', type=str, default='dataset/test.json')
    parser.add_argument('--val_path', type=str, default='dataset/val.json')'''
    parser.add_argument('--model_path', type=str, default='model.pth')
  
    args = parser.parse_args()
    main(args)
 
# python transformer_kan.py --mode "train" --em_model_name "bert-base-cased" --ds_name "mrpc" --epochs 5 --batch_size 4 --max_len 512 --n_size 1 --m_size 768 --n_hidden 64 --n_class 2

# python transformer_kan.py --mode "test" --model_name "facebook/bart-base" --model_path "bart-base\checkpoint-1321" --test_path "dataset/test.json" --test_batch_size 4 --max_source_length 256 --min_target_length 1