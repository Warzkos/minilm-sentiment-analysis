import torch
torch.cuda.is_available()

import torch
device = torch.device('cuda')

import warnings
warnings.filterwarnings("ignore")
from datetime import datetime as dt
from transformers import AutoTokenizer, pipeline, AutoModelForSequenceClassification

miniLM_L12='microsoft/Multilingual-MiniLM-L12-H384'
miniLMv2_L12='nreimers/mMiniLMv2-L12-H384-distilled-from-XLMR-Large'
miniLMv2_L6='nreimers/MiniLMv2-L6-H384-distilled-from-RoBERTa-Large'

model_l3 = AutoModelForSequenceClassification.from_pretrained(miniLM_L12)
model_l3.to(device)
tokenizer1 = AutoTokenizer.from_pretrained(miniLM_L12)

model_l6 = AutoModelForSequenceClassification.from_pretrained(miniLMv2_L12)
model_l6.to(device)
tokenizer2 = AutoTokenizer.from_pretrained(miniLMv2_L12)

model_l6v2 = AutoModelForSequenceClassification.from_pretrained(miniLMv2_L6)
model_l6v2.to(device)
tokenizer3 = AutoTokenizer.from_pretrained(miniLMv2_L6)

"""# Załaduj dane"""

import pandas as pd
import datasets
from datasets import Dataset


def full_preprocess(df_patch1, df_patch2):
  en_train = pd.read_xml(df_patch1)
  en_test = pd.read_xml(df_patch1)
  en_full = pd.concat([en_train, en_test], ignore_index=True)
  for i in range(len(en_full)):
    if en_full.loc[i, 'rating'] <= 3.0:
      en_full.at[i, 'rating'] = int(0)
    elif en_full.at[i, 'rating'] > 3.0:
      en_full.at[i, 'rating'] = int(1)
  #print(en_full)
  x = en_full[['text']].copy()
  y = en_full[['rating']].copy()
  return x, y

def preprocess(df_patch1):
  en_full = pd.read_xml(df_patch1)
  for i in range(len(en_full)):
    if en_full.loc[i, 'rating'] <= 3.0:
      en_full.at[i, 'rating'] = int(0)
    elif en_full.at[i, 'rating'] > 3.0:
      en_full.at[i, 'rating'] = int(1)
  out = Dataset.from_pandas(en_full)
  return out

en_train = preprocess("/home/s249403/projects/sentiment_multilingual/cls-acl10-unprocessed/en/books/train.review")
en_test = preprocess("/home/s249403/projects/sentiment_multilingual/cls-acl10-unprocessed/en/books/test.review")
jp_train = preprocess("/home/s249403/projects/sentiment_multilingual/cls-acl10-unprocessed/jp/books/train.review")
jp_test = preprocess("/home/s249403/projects/sentiment_multilingual/cls-acl10-unprocessed/jp/books/test.review")
fr_train = preprocess("/home/s249403/projects/sentiment_multilingual/cls-acl10-unprocessed/fr/books/train.review")
fr_test = preprocess("/home/s249403/projects/sentiment_multilingual/cls-acl10-unprocessed/fr/books/test.review")
de_train = preprocess("/home/s249403/projects/sentiment_multilingual/cls-acl10-unprocessed/de/books/train.review")
de_test = preprocess("/home/s249403/projects/sentiment_multilingual/cls-acl10-unprocessed/de/books/test.review")

X_en, y_en = full_preprocess("/home/s249403/projects/sentiment_multilingual/cls-acl10-unprocessed/en/books/train.review","/home/s249403/projects/sentiment_multilingual/cls-acl10-unprocessed/en/books/test.review")
x_jp, y_jp = full_preprocess("/home/s249403/projects/sentiment_multilingual/cls-acl10-unprocessed/jp/books/train.review","/home/s249403/projects/sentiment_multilingual/cls-acl10-unprocessed/jp/books/test.review")
x_fr, y_fr = full_preprocess("/home/s249403/projects/sentiment_multilingual/cls-acl10-unprocessed/fr/books/train.review","/home/s249403/projects/sentiment_multilingual/cls-acl10-unprocessed/fr/books/test.review")
x_de, y_de = full_preprocess("/home/s249403/projects/sentiment_multilingual/cls-acl10-unprocessed/de/books/train.review","/home/s249403/projects/sentiment_multilingual/cls-acl10-unprocessed/de/books/test.review")

XX = {'X_en': X_en, 'X_jp': x_jp, 'X_fr': x_fr, 'X_de': x_de, }
yy = {'y_en': y_en, 'y_jp': y_jp, 'y_fr': y_fr, 'y_de': y_de, }

"""# Trenuj wszystkie"""

import numpy as np
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader
from transformers import AdamW
import copy
from datasets import load_dataset
from torch.utils.data import Dataset as DS

class reviewDataset(DS):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

folds1 = StratifiedKFold(n_splits=5)
folds2 = StratifiedKFold(n_splits=5)
folds3 = StratifiedKFold(n_splits=5)
folds4 = StratifiedKFold(n_splits=5)

skf = StratifiedKFold(n_splits=5)
num_epochs = 3
#model= AutoModelForSequenceClassification.from_pretrained(miniLMv2_L6)
#model.to(device)
#tokenizer = AutoTokenizer.from_pretrained(miniLMv2_L6)

#splits1 = folds1.split(X_val, y_val)
#splits2 = folds2.split(x_jp, y_jp)
#splits3 = folds3.split(x_fr, y_fr)
#splits3 = folds4.split(x_de, y_de)
#print(X_val)

from copy import deepcopy as dc

models={'miniLM_L12': model_l3, 'miniLMv2_L12': model_l6, 'miniLMv2_L6': model_l6v2,}
tokenizers={'miniLM_L12':tokenizer1, 'miniLMv2_L12':tokenizer2, 'miniLMv2_L6':tokenizer3,}
for (X_key, X_val), (y_key, y_val) in zip(XX.items(), yy.items()):
  #print('#\t'*100)
  #print(f'{X_key}, {y_key}')
  #for (X_key, X_val), (y_key, y_val) in zip(dict1.items(), dict2.items()):
  for name, model in models.items():
    model = dc(model)
    best_acc = 0
    #print('-\t'*60)
    print(f'model name: {name}, dataset: {X_key}_{y_key}')
    fold_cnt = 0
    start_time = dt.now()
    for train_index, test_index in skf.split(X_val, y_val):
        #print(f'fold: {fold_cnt}')
        fold_cnt+=1
        batch_size = 32
        X_train_fold, X_test_fold = Dataset.from_pandas(X_val.loc[X_val.index[train_index]]), Dataset.from_pandas(X_val.loc[X_val.index[test_index]])
        y_train_fold, y_test_fold = Dataset.from_pandas(y_val.loc[X_val.index[train_index]]), Dataset.from_pandas(y_val.loc[X_val.index[test_index]])

        
        train_data = np.array(X_train_fold['text']).tolist()
        train_labels = np.array(y_train_fold['rating']).tolist()
        test_data = np.array(X_test_fold['text']).tolist()
        test_labels = np.array(y_test_fold['rating']).tolist()
        train_encodings = tokenizers[name](train_data, truncation=True, padding=True, max_length=384, return_tensors="pt")
        test_encodings = tokenizers[name](test_data, truncation=True, padding=True, max_length=384, return_tensors="pt")
        train_labels = [int(x) for x in train_labels]
        test_labels = [int(x) for x in test_labels]
        

        train_dataset = reviewDataset(train_encodings, train_labels)
        test_dataset = reviewDataset(test_encodings, test_labels)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        optim = AdamW(model.parameters(), lr=5e-5)        
        epoch_cnt = 0
        occ_loss = 0
        fold_loss = []
        for epoch in range(num_epochs):
            model.train()
            #print('epoch_cnt='+ str(epoch_cnt))
            epoch_cnt+=1
            #print('train')
            for batch in train_loader:
                optim.zero_grad()
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                input_ids=input_ids.squeeze()

                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs[0]
                fold_loss.append(loss.item())
                loss.backward()
                optim.step()
            #print('eval')
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
                accuracy = acc/occ
                precision = tp/(tp+fp) if tp+fp!=0 else -1
                recall = tp/(tp+fn) if tp+fn!=0 else -1
                f1 = 2*precision*recall/(precision+recall) if tp+fp!=0 and tp+fn!=0 else -1
                true_negative_rate = tn/(tn+fp) if tn+fp!=0 else -1
                print(f'fold: {fold_cnt-1}, epoch: {epoch_cnt-1}')
                print(f'acc: {accuracy:.3f}, pre: {precision:.3f}, rec: {recall:.3f}, f1s: {f1:.3f}, tnr: {true_negative_rate:.3f}, tp: {tp}, tn: {tn}, fp: {fp}, fn: {fn}, sum: {tp+tn+fp+fn}')
                if (accuracy>best_acc):
                  torch.save(model.state_dict(), '/home/s249403/projects/sentiment_multilingual/models/'+'best_'+name+'_'+X_key+'_'+y_key+'.pt') 
                  best_acc = acc
    end_time=dt.now()
    print(f'start_time: {start_time.time()}, end_time: {end_time.time()}, time_diff: {end_time-start_time}')