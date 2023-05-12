import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
import numpy as np
from torch.utils.data import Dataset, DataLoader

batch_size = 50

print('import model')
device = torch.device('cuda')
model = AutoModelForSequenceClassification.from_pretrained('nreimers/MiniLM-L6-H384-uncased')

path = '/home/s249403/projects/sentiment/sentiment_model_L6_fold_5.pt'
print('load model ' + str(path))
model.load_state_dict(torch.load(path))
model.to(device)
model.eval()

tokenizer = AutoTokenizer.from_pretrained('nreimers/MiniLM-L6-H384-uncased')

print('loading dataset')
#imdb dataset
dataset = load_dataset("imdb", split='test')

test_texts = np.array(dataset['text']).tolist()
test_labels = np.array(dataset['label']).tolist()

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

print('embedding')
# embeddings
test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=384, return_tensors="pt")
test_dataset = imdbDataset(test_encodings, test_labels)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

print('begin eval')
with torch.no_grad():
    acc = 0
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    occ = 0
    for batch in test_loader:
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
