import torch
device = torch.device('cuda')

from transformers import AutoTokenizer, pipeline, AutoModelForSequenceClassification
from datasets import load_dataset
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from transformers import AdamW
from sklearn.model_selection import StratifiedKFold
import copy
import csv
import time

model_name='microsoft/MiniLM-L12-H384-uncased'

model = AutoModelForSequenceClassification.from_pretrained(model_name)
model.to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name)

dataset = load_dataset("imdb", split='train')

print(len(dataset))

count = 0

for i in range(len(dataset)):
    if dataset[i]['label'] == 1:
        count += 1

print('c0=' + str(len(dataset) - count) + ', c1=' + str(count))

X = np.array(dataset['text'])
y = np.array(dataset['label'])


class imdbDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


 # if torch.cuda.is_available() else torch.device('cpu')
num_epochs = 3
batch_size = 40
k_folds= 5
print('batch size= ' + str(batch_size))
print('num of epochs per fold= '+str(num_epochs))
print('num of folds= '+str(k_folds))
skf = StratifiedKFold(n_splits=k_folds, random_state=1234, shuffle=True)
loss_table=[]
ts = time.time()
for i, (train_index, test_index) in enumerate(skf.split(X, y)):
    print('fold' + str(i + 1))
    model.cuda()

    train_texts = X[train_index].tolist()
    train_labels = y[train_index].tolist()
    test_texts = X[test_index].tolist()
    test_labels = y[test_index].tolist()

    train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=384, return_tensors="pt")
    test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=384, return_tensors="pt")

    train_dataset = imdbDataset(train_encodings, train_labels)
    test_dataset = imdbDataset(test_encodings, test_labels)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    optim = AdamW(model.parameters(), lr=5e-5)
    print('begin')
    epoch_cnt = 0
    occ_loss = 0
    fold_loss = []
    for epoch in range(num_epochs):
        print('train')
        model.train()
        batch_cnt = 0
        print('epoch_cnt='+ str(epoch_cnt))
        epoch_cnt+=1
        for batch in train_loader:
            if batch_cnt % (10000 // batch_size) == 0:
                print('batch_cnt='+ str(batch_cnt))
            batch_cnt+=1
            optim.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)

            loss = outputs[0]
            fold_loss.append(loss.item())
            loss.backward()
            optim.step()
        print('eval')
        model.eval()
        val_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
        with torch.no_grad():
            acc = 0
            tp = 0
            tn = 0
            fp = 0
            fn = 0
            occ = 0
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                labels = batch['labels'].to(device)

                target = model(input_ids, labels=labels)
                table = torch.nn.functional.softmax(target['logits'], dim=1)
                eval_arr=[]
                for prob in table:
                    if prob[0]<0.5:
                        eval_arr.append(1)
                    else:
                        eval_arr.append(0)
                for p, l in zip(eval_arr, labels):
                    if p == l:
                        acc+=1
                        if p == 1:
                            tp+=1
                        else:
                            tn+=1
                    if p==1 and l==0:
                        fp+=1
                    if l==1 and p==0:
                        fn+=1
                    occ+=1
            prec = tp/(tp+fp)
            rec = tp/(tp+fn)
            print('validation acc= ' + str(acc/occ))
            print('validation prec='+str(prec))
            print('validation rec= '+str(rec))
            print('validation f1=  '+str(2*prec*rec/(prec+rec)))
            print('validation tnr= '+ str(tn/(tn+fp)))

    loss_table.append(fold_loss)
    torch.save(model.state_dict(), '/home/s243425/sentiment/sentiment_model_L12_bs='+str(batch_size)+'_fold_'+str(i+1)+'.pt')

te = time.time()
print('duration[s] = ' + str(te - ts))

with open(f"output_bs={batch_size}.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerows(loss_table)