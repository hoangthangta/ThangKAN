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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = load_dataset('glue', 'wnli')
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
test_labels = [1  for item in test_set]

all_data = train_data + val_data + test_data
all_labels = train_labels + val_labels + test_labels
df_labels = pd.DataFrame(all_labels, columns=['label'])

# BOW
n_features = 30
vectorizer = CountVectorizer(max_features = n_features*n_features) # 20*20
BOW = vectorizer.fit_transform(all_data)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(BOW, np.asarray(df_labels), test_size = 217, train_size = 635, random_state=0)
x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, train_size = 71, random_state=0)

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
trainloader = DataLoader(final_train, batch_size=4, shuffle=True)
valloader = DataLoader(final_val, batch_size=4, shuffle=False)
testloader = DataLoader(final_test, batch_size=4, shuffle=False)

i = 0 
for item in trainloader:
    print(item[0].size())
    if (i == 0): break

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
for epoch in range(50):
    # Train
    model.train()
    train_loss = 0
    train_accuracy = 0
    with tqdm(trainloader) as pbar:
        for i, (texts, labels) in enumerate(pbar):
            texts = texts.view(-1, n_features*n_features).to(device)
            #print('texts: ', texts)
            optimizer.zero_grad()
            output = model(texts)
            loss = criterion(output, labels.to(device))
            train_loss += loss.item()
            loss.backward()
            optimizer.step()
            accuracy = (output.argmax(dim=1) == labels.to(device)).float().mean()
            train_accuracy += accuracy.item()
            
            pbar.set_postfix(train_loss=train_loss/len(trainloader), train_accuracy=train_accuracy/len(trainloader), lr=optimizer.param_groups[0]['lr'])

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