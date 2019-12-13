import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import pickle
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from pytorch_pretrained_bert import BertTokenizer, BertConfig
from pytorch_pretrained_bert import BertAdam, BertForSequenceClassification, BertForQuestionAnswering

from tqdm import tqdm, trange
import pandas as pd
import io
import numpy as np
import matplotlib.pyplot as plt
import sys
np.set_printoptions(threshold=sys.maxsize)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()
print(torch.cuda.get_device_name(0)+" "+str(n_gpu))

queries=[]
paragraphs=[]
labels=[]
data=open("./train_data/data.txt")
print("file opened")
unrel=0
rel=0
query=""
paragraph=""
label=-1
i=0
for line in data:
  if i%4==0:
    query=line
  elif i%4==1:
    paragraph=line
  elif i%4==2:
    label=int(line)
    if label==1:
      queries.append(query)
      paragraphs.append(paragraph)
      labels.append(label)
      rel+=1
    elif label==0:
      queries.append(query)
      paragraphs.append(paragraph)
      labels.append(label)
      unrel+=1
  i+=1
print(str(rel)+" relevent pairs")
print(str(unrel)+" not relevent pairs")
print(str(rel+unrel)+" train data loaded")

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

tokenized_queries = [tokenizer.tokenize(query) for query in queries]
tokenized_paragraph = [tokenizer.tokenize(paragraph) for paragraph in paragraphs]
segment_mask=[]
tokenized_texts=[]
for i in range(0,len(tokenized_queries)):
    query_seg_mask=[0 for qt in tokenized_queries[i]]
    para_seg_mask=[1 for pt in tokenized_paragraph[i]]
    segment_mask.append([0] + query_seg_mask + [0] + para_seg_mask + [1])
    tokenized_texts.append(['[CLS]'] + tokenized_queries[i] + ['[SEP]'] + tokenized_paragraph[i] + ['[SEP]'])
    
MAX_LEN = 128 #for padding and making input size constant
input_ids = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_texts]
input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")
segment_mask = pad_sequences(segment_mask, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")
attention_masks = []
for seq in input_ids:
  seq_mask = [float(i>0) for i in seq]
  attention_masks.append(seq_mask)

train_inputs=input_ids
train_labels=labels
train_masks=attention_masks
train_seg_masks=segment_mask

train_inputs = torch.tensor(train_inputs)
train_labels = torch.tensor(train_labels)
train_masks = torch.tensor(train_masks)
train_seg_masks = torch.tensor(train_seg_masks)

batch_size = 32

train_data = TensorDataset(train_inputs, train_masks, train_seg_masks, train_labels)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
model.cuda()

param_optimizer = list(model.named_parameters())
no_decay = ['bias', 'gamma', 'beta']
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
     'weight_decay_rate': 0.01},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
     'weight_decay_rate': 0.0}
]

optimizer = BertAdam(optimizer_grouped_parameters, lr=2e-5, warmup=.1)

############################################fine tuning phase starts here######################################

def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

train_loss_set = []

epochs = 4

loop=0
for _ in trange(epochs, desc="Epoch"):
  loop+=1
  
  model.train()
  
  tr_loss = 0
  nb_tr_examples, nb_tr_steps = 0, 0
  
  for step, batch in enumerate(train_dataloader):
    batch = tuple(t.to(device) for t in batch)

    b_input_ids, b_input_mask, b_seg_mask, b_labels = batch

    optimizer.zero_grad()

    loss = model(b_input_ids, token_type_ids=b_seg_mask, attention_mask=b_input_mask, labels=b_labels)
    train_loss_set.append(loss.item())    
    
    loss.backward()

    optimizer.step()
    
    tr_loss += loss.item()
    nb_tr_examples += b_input_ids.size(0)
    nb_tr_steps += 1

  print("Train loss: {}".format(tr_loss/nb_tr_steps))
    
  pickle.dump(model, open('seventh'+str(loop)+'.model', 'wb'))

plt.figure(figsize=(15,8))
plt.title("Training loss")
plt.xlabel("Batch")
plt.ylabel("Loss")
plt.plot(train_loss_set)
plt.savefig('train_loss_seventh.png')