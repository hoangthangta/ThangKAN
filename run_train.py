from datasets import load_dataset
from file_io import *
from kan import KAN
from models import EfficientKAN, TransformerClassifier, TransformerMLP

from pathlib import Path
#from sklearn.preprocessing import normalize
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader
from tqdm import tqdm

import argparse
import gc
import numpy as np
import pandas as pd
import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim

from utils import *

def create_data_loader(ds_name, dataset, tokenizer, max_len = 512, batch_size = 4, shuffle = False):

    texts, labels = [], []
    for item in dataset:
       
        text = ''
        if (ds_name in ['mrpc', 'rte', 'wnli']):
            # only for BERT, other models may have different special tokens
            text = '[CLS] ' + item['sentence1'] + ' [SEP] ' + item['sentence2'] + ' [SEP]'
 
        if (ds_name == 'cola'):
            text = item['sentence']
        texts.append(text)
        
        try: labels.append(item['label'])
        except: labels.append(-1) 
    
    ds = TextDataset(texts= np.array(texts),
                         labels = np.array(labels),
                         tokenizer=tokenizer,
                         max_len=max_len)
                         
    return DataLoader(ds, batch_size=batch_size, num_workers=1, shuffle = shuffle)
    
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

def get_embeddings(data, n_size = 1, m_size = 768, embed_type = 'pool'):
    
    with torch.no_grad():
        input_ids = data["input_ids"].to(device)
        attention_mask = data["attention_mask"].to(device)

        embeddings = {}
        if (embed_type == 'hidden'): # last hidden state, 512 x 768 for BERT
            outputs = em_model(input_ids=input_ids, attention_mask=attention_mask)
            embeddings = outputs['last_hidden_state']
            embeddings = torch.sum(embeddings, (1), keepdim = True) # require memory 
            embeddings = embeddings.view(-1, n_size*m_size)
            
        elif (embed_type == 'weight'): # weight, 512 x 768 for BERT
            embedding_matrix = model.embeddings.word_embeddings.weight
            embeddings = embedding_matrix[input_ids]
            embeddings = torch.sum(embeddings, (1), keepdim = True) # require memory 
            embeddings = embeddings.view(-1, n_size*m_size)
            
        else: # pool
            outputs = em_model(input_ids=input_ids, attention_mask=attention_mask)
            embeddings = outputs[-1]
            
            # normalize
            #std_mean = torch.std_mean(embeddings, dim=1, keepdim=True)
            #embeddings = (embeddings - std_mean[1])/std_mean[0]

        del outputs, input_ids, attention_mask
        return embeddings


def train_model(trainloader, valloader, network = 'classifier', ds_name = 'mrpc', em_model_name = 'bert-base-cased', \
                        epochs = 20, n_size = 1, m_size = 768, n_hidden = 64, n_class = 2, embed_type = 'pool'):
    """
        for training classifier, efficientkan, and mlp
    """
    
    start = time.time()
    
    model = {}
    if (network == 'classifier'):
        model = TransformerClassifier(n_class, em_model_name)
        model.to(device)
    elif(network == 'efficientkan'):
        model = EfficientKAN([n_size*m_size, n_hidden, n_class])  # grid=5, k=3
        model.to(device)
    elif(network == 'mlp'):
        model = TransformerMLP(n_size*m_size, [n_hidden], n_class, em_model_name)
        model.to(device)
    elif(network == 'kan'):
        # It takes a very long time to infer an output from the original KAN package
        model = KAN(width=[n_size*m_size, n_hidden, n_class], grid=5, k=3, device = device)
        #model.to(device)
    else:
        print("Please choose --network parameter as one of ('classifier', efficientkan, 'mlp')!")
    
    #count_parameters(model)    
    # define optimizer
    #optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3) # 1e-5, 2e-5
    
    # define learning rate scheduler
    #scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8)
    total_steps = len(trainloader) * epochs
    scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=total_steps
        )
        
    # define loss
    criterion = nn.CrossEntropyLoss()
    
    best_accuracy = 0
    best_epoch = 0
    
    # create model saved name 
    model_id = ''
    try: model_id = em_model_name.split('/')[1]
    except: model_id = em_model_name 
    
    output_path = 'output/' + model_id 
    Path(output_path).mkdir(parents=True, exist_ok=True)
    saved_model_name =  model_id + '_' +  ds_name + '_'+ network + '.pth'
    saved_model_history =  model_id + '_' +  ds_name + '_' + network + '.json'
    with open(os.path.join(output_path, saved_model_history), 'w') as fp: pass
    
    
    for epoch in range(epochs):
        # train
        if (network != 'kan'): model.train()
        em_model.train() 
        train_loss = 0
        train_accuracy = 0
        with tqdm(trainloader) as pbar:
            for i, items in enumerate(pbar):
                optimizer.zero_grad()
                labels = items['label']
                
                outputs = {}
                if (network in ['classifier', 'mlp']):
                    input_ids = items["input_ids"].to(device)
                    attention_mask = items["attention_mask"].to(device)
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                elif(network == 'efficientkan'):
                    texts = get_embeddings(items, n_size = n_size, m_size = m_size, embed_type = embed_type).to(device)
                    outputs = model(texts.to(device))
                elif(network == 'kan'): 
                    # embed_type always 'pool'
                    texts = get_embeddings(items, n_size = n_size, m_size = m_size, embed_type = 'pool').to(device)
                    texts = reduce_size(texts, n_size = n_size, m_size = m_size)
                    outputs = model(texts.to(device))              
                else:
                    print("Please choose --network parameter as one of ('classifier', efficientkan, 'mlp')!")

                loss = criterion(outputs, labels.to(device))
                train_loss += loss.item()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()

                train_accuracy += ((outputs.argmax(dim=1) == labels.to(device)).float().mean().item())
                pbar.set_postfix(train_loss=train_loss/len(trainloader), train_accuracy=train_accuracy/len(trainloader), lr=optimizer.param_groups[0]['lr'])     

        train_loss /= len(trainloader)
        train_accuracy /= len(trainloader)
        
        # update learning rate
        #scheduler.step() 
        
        # validation
        if (network != 'kan'): model.eval()
        em_model.eval() 
        val_loss = 0
        val_accuracy = 0
        
        with torch.no_grad():
            with tqdm(valloader) as pbar:
                for i, items in enumerate(pbar):
                    labels = items['label']
                    
                    outputs = {}
                    if (network in ['classifier', 'mlp']):
                        input_ids = items["input_ids"].to(device)
                        attention_mask = items["attention_mask"].to(device)
                        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    elif(network == 'efficientkan'):
                        texts = get_embeddings(items, n_size = n_size, m_size = m_size, embed_type = embed_type).to(device)
                        outputs = model(texts.to(device))
                    elif(network == 'kan'): 
                        # embed_type always 'pool'
                        texts = get_embeddings(items, n_size = n_size, m_size = m_size, embed_type = 'pool').to(device)
                        texts = reduce_size(texts, n_size = n_size, m_size = m_size)
                        outputs = model(texts.to(device))     
                    else:
                        print("Please choose --network parameter as one of ('classifier', efficientkan, 'mlp')!")
                    
                    val_loss += criterion(outputs, labels.to(device)).item()
                    val_accuracy += ((outputs.argmax(dim=1) == labels.to(device)).float().mean().item())
                    pbar.set_postfix(val_loss=val_loss/len(valloader), val_accuracy=val_accuracy/len(valloader))
        val_loss /= len(valloader)
        val_accuracy /= len(valloader)
        
        if (val_accuracy > best_accuracy):
            best_accuracy = val_accuracy
            best_epoch = epoch
            if (network == 'kan'):
                torch.save(model.state_dict(), output_path + '/' + saved_model_name)
            else:
                torch.save(model, output_path + '/' + saved_model_name)
        
        write_single_dict_to_jsonl_file(output_path + '/' + saved_model_history, {'epoch':epoch+1, 'val_accuracy':val_accuracy, 'train_accuracy':train_accuracy, 'best_accuracy': best_accuracy, 'best_epoch':best_epoch+1, 'val_loss': val_loss, 'train_loss':train_loss}, file_access = 'a')
          
        print(f"Epoch {epoch + 1}, Train Loss: {train_loss}, Train Accuracy: {train_accuracy}")
        print(f"Epoch {epoch + 1}, Val Loss: {val_loss}, Val Accuracy: {val_accuracy}")
        
        torch.cuda.empty_cache()
        gc.collect()
    
    end = time.time()
    
    write_single_dict_to_jsonl_file(output_path + '/' + saved_model_history, {'training time':end-start}, file_access = 'a')              

def infer_model(test_loader, network = 'classifier', model_path = 'model.pth', embed_type = 'pool', n_size = 1, m_size = 768):

    #model = TheModelClass(*args, **kwargs)
    model = EfficientKAN([768, 64, 2])  # grid=5, k=3
    #model.load_state_dict(torch.load(model_path))
    #model.eval()
    model = torch.load(model_path)
    #model.eval()
    
    #print('model parameters: ', count_parameters(model))
    
    #return
    criterion = nn.CrossEntropyLoss()
    test_loss = 0
    test_accuracy = 0
    with torch.no_grad():
        with tqdm(test_loader) as pbar:
            for i, items in enumerate(pbar):
                labels = items['label']
                
                outputs = {}
                if (network in ['classifier', 'mlp']):
                    input_ids = items["input_ids"].to(device)
                    attention_mask = items["attention_mask"].to(device)
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                elif(network == 'efficientkan'):
                    texts = get_embeddings(items, n_size = n_size, m_size = m_size, embed_type = embed_type).to(device)
                    outputs = model(texts.to(device))
                elif(network == 'kan'): 
                    # embed_type always 'pool'
                    texts = get_embeddings(items, n_size = n_size, m_size = m_size, embed_type = 'pool').to(device)
                    texts = reduce_size(texts, n_size = n_size, m_size = m_size)
                    outputs = model(texts.to(device)) 
                else:
                    print("Please choose --network parameter as one of ('classifier', efficientkan, 'mlp')!")
            
                test_loss += criterion(outputs, labels.to(device)).item()
                test_accuracy += ((outputs.argmax(dim=1) == labels.to(device)).float().mean().item())
                pbar.set_postfix(test_loss=test_loss.item(), test_accuracy=test_accuracy.item())

        test_loss /= len(test_loader)
        test_accuracy /= len(test_loader)

    print(test_accuracy, test_loss)
    return {'accuracy':test_accuracy, 'avg_loss':test_loss}

def prepare_dataset(ds_name = 'mrpc'):
    
    dataset = load_dataset('glue', ds_name)
    return dataset

def build_data_loader(ds_name, em_model_name, max_len = 512, batch_size = 4, test_only = False):
    
    dataset = prepare_dataset(ds_name)
    print('First example :', dataset['train'][0])
    print('First example :', dataset['validation'][0])
    
    tokenizer = AutoTokenizer.from_pretrained(em_model_name)
    
    global em_model
    em_model = AutoModel.from_pretrained(em_model_name)
    em_model = em_model.to(device)
    
    train_loader, val_loader, test_loader = [], [], []
    test_loader = create_data_loader(ds_name, dataset['test'], tokenizer, max_len = max_len, batch_size = batch_size)
    if (test_only == False):
        train_loader = create_data_loader(ds_name, dataset['train'], tokenizer, max_len = max_len, \
                                            batch_size = batch_size, shuffle = False)
        val_loader = create_data_loader(ds_name, dataset['validation'], tokenizer, \
                                            max_len = max_len, batch_size = batch_size)
    
    return {'train': train_loader, 'validation': val_loader, 'test': test_loader}

def main(args):
    if (args.mode == 'train'):
        loader = build_data_loader(ds_name = args.ds_name, em_model_name = args.em_model_name, \
                                        max_len = args.max_len, batch_size = args.batch_size)
       
        train_model(loader['train'], loader['validation'], network = args.network, ds_name = args.ds_name, \
                    em_model_name = args.em_model_name, epochs = args.epochs, n_size = args.n_size, \
                    m_size = args.m_size, n_hidden = args.n_hidden, n_class = args.n_class, embed_type = args.embed_type)

    elif (args.mode == 'test'):
        
        loader = build_data_loader(ds_name = args.ds_name, em_model_name = args.em_model_name, max_len = args.max_len, batch_size = args.batch_size)
        
        # GLUE datasets have no "test set" with labels, so we use "validation set" instead.
        infer_model(loader['validation'], network = args.network, model_path = args.model_path, embed_type = args.embed_type, n_size = args.n_size, m_size = args.m_size)
        
       
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Training Parameters')
    parser.add_argument('--mode', type=str, default='train') # or test
    parser.add_argument('--network', type=str, default='kan') # or classifier
    parser.add_argument('--em_model_name', type=str, default='bert-base-cased') # or test
    parser.add_argument('--ds_name', type=str, default='mrpc')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--max_len', type=int, default=512)
    
    parser.add_argument('--n_size', type=int, default=1)
    parser.add_argument('--m_size', type=int, default=768)
    parser.add_argument('--n_hidden', type=int, default=64)
    parser.add_argument('--n_class', type=int, default=2)
    parser.add_argument('--embed_type', type=str, default='pool') # only for KAN
    parser.add_argument('--device', type=str, default='cuda')
    '''parser.add_argument('--train_path', type=str, default='dataset/train.json') 
    parser.add_argument('--test_path', type=str, default='dataset/test.json')
    parser.add_argument('--val_path', type=str, default='dataset/val.json')'''
    parser.add_argument('--model_path', type=str, default='model.pth')
    args = parser.parse_args()
    
    global device
    device = args.device
    if (args.device == 'cuda'): # check available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    main(args)
    
# ['rte', 'wnli', 'mrpc', 'cola']
#python run_train.py --mode "train" --network "kan" --em_model_name "bert-base-cased" --ds_name "wnli" --epochs 10 --batch_size 4 --max_len 512 --n_size 1 --m_size 8 --n_hidden 64 --n_class 2 --device "cpu"

#python run_train.py --mode "train" --network "efficient kan" --em_model_name "bert-base-cased" --ds_name "wnli" --epochs 10 --batch_size 4 --max_len 512 --n_size 1 --m_size 768 --n_hidden 64 --n_class 2 --embed_type "pool"

#python run_train.py --mode "test" --network "efficientkan" --em_model_name "bert-base-cased" --ds_name "wnli" --batch_size 4 --max_len 512  --model_path "output/bert-base-cased/bert-base-cased_wnli_kan.pth" --embed_type "pool"

