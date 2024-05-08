from efficient_kan import KAN

import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torchtext
#import torchvision
#import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.feature_extraction.text import CountVectorizer
from datasets import load_dataset
from sklearn.preprocessing import normalize

from transformers import BertModel, BertTokenizer



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = load_dataset('glue', 'mrpc')
train_set = dataset['train']
val_set = dataset['validation']
test_set = dataset['test']
print(len(train_set), len(val_set), len(test_set)) # 3668 408 1725

# concat 2 sentences
train_data = [item['sentence1'] + ' ### ' + item['sentence2']  for item in train_set]
train_labels = [item['label']  for item in train_set]

val_data = [item['sentence1'] + ' ### ' + item['sentence2']  for item in val_set]
val_labels = [item['label']  for item in val_set]

test_data = [item['sentence1'] + ' ### ' + item['sentence2']  for item in test_set]
test_labels = [item['label']  for item in test_set]

all_data = train_data + val_data + test_data
all_labels = train_labels + val_labels + test_labels
df_labels = pd.DataFrame(all_labels, columns=['label'])


# BERT
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
model = BertModel.from_pretrained('bert-base-cased')

def create_data_loader(dataset, tokenizer, max_len, batch_size = 4):

    ds = PreparedDataset(texts=np.array(dataset),
                         tokenizer=tokenizer,
                         max_len=max_len)
    return DataLoader(ds, batch_size=batch_size, num_workers=4)
    
class PreparedDataset():
    def __init__(self, texts, tokenizer, max_len):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = str(self.texts[item])
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
                'attention_mask': encoding['attention_mask'].flatten()
            }


all_data_loader = create_data_loader(all_data, tokenizer, 512, 4)

i = 0
emb_data = []
for d in all_data_loader:
        
    sys.stdout.write('Training batch: %d/%d \r' % (i, len(data_loader)))
    #sys.stdout.flush()
        
    i = i + 1
    input_ids = d["input_ids"].to(device)
    attention_mask = d["attention_mask"].to(device)
    categories = d["categories"].to(device)
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    emb_data += outputs['last_hidden_state']

print('emb_data: ', len(emb_data))
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

''''
# BOW
n_features = 100
vectorizer = CountVectorizer(max_features = n_features*n_features) # 20*20
BOW = vectorizer.fit_transform(all_data)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(BOW, np.asarray(df_labels), test_size = 2133, train_size = 3668, random_state=0)
x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, train_size = 408, random_state=0)

print('train shape: ', x_train.shape, y_train.shape)
print('val shape: ', x_val.shape, y_val.shape)
print('test shape: ', x_test.shape, y_test.shape)

# concat data + labels
#x.toarray().astype(np.float32), torch.from_numpy([x.toarray().astype(np.float32)])

final_train = []
for x, y in zip(x_train, y_train): final_train.append( (x.toarray().astype(np.float32).reshape(1, n_features, n_features), int(y)) )

final_val = []
for x, y in zip(x_val, y_val): final_val.append( (x.toarray().astype(np.float32).reshape(1, n_features, n_features), int(y)) )

final_test = []
for x, y in zip(x_test, y_test): final_test.append( (x.toarray().astype(np.float32).reshape(1, n_features, n_features), int(y)) )

# load data
trainloader = DataLoader(final_train, batch_size=64, shuffle=True)
valloader = DataLoader(final_val, batch_size=64, shuffle=False)
testloader = DataLoader(final_test, batch_size=64, shuffle=False)

i = 0 
for item in trainloader:
    print(item[0].size())
    if (i == 0): break
'''

# define KAN model
model = KAN([n_features*n_features, 64, 2])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
# define optimizer
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
# define learning rate scheduler
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8)

# define loss
criterion = nn.CrossEntropyLoss()
best_accuracy = 0
for epoch in range(10):
    # Train
    model.train()
    with tqdm(trainloader) as pbar:
        for i, (texts, labels) in enumerate(pbar):
            texts = texts.view(-1, n_features*n_features).to(device)
            optimizer.zero_grad()
            output = model(texts)
            loss = criterion(output, labels.to(device))
            loss.backward()
            optimizer.step()
            accuracy = (output.argmax(dim=1) == labels.to(device)).float().mean()
            pbar.set_postfix(loss=loss.item(), accuracy=accuracy.item(), lr=optimizer.param_groups[0]['lr'])

    # validation
    model.eval()
    val_loss = 0
    val_accuracy = 0
    with torch.no_grad():
        for texts, labels in valloader:
            texts = texts.view(-1, n_features*n_features).to(device)
            output = model(texts)
            val_loss += criterion(output, labels.to(device)).item()
            val_accuracy += (
                (output.argmax(dim=1) == labels.to(device)).float().mean().item()
            )
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

'''model = torch.load('model.pth')
model.eval()

criterion = nn.CrossEntropyLoss()
test_loss = 0
test_accuracy = 0
with torch.no_grad():
    for texts, labels in testloader:
        texts = texts.view(-1, n_features*n_features).to(device)
        output = model(texts)
        test_loss += criterion(output, labels.to(device)).item()
        test_accuracy += ((output.argmax(dim=1) == labels.to(device)).float().mean().item())

    test_loss /= len(testloader)
    test_accuracy /= len(testloader)

print(test_accuracy, test_loss)'''
