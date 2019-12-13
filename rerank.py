import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import math
import collections
import operator
import pickle
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from pytorch_pretrained_bert import BertTokenizer, BertConfig
from pytorch_pretrained_bert import BertAdam, BertForSequenceClassification
from tqdm import tqdm, trange
import pandas as pd
import io
import numpy as np
import matplotlib.pyplot as plt
import sys
from sklearn.metrics import confusion_matrix
np.set_printoptions(threshold=sys.maxsize)

def round_nearest(x, a):
    return round(x*1.0 / a) * a

def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

def flat_confusion(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return confusion_matrix(labels_flat, pred_flat)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()
print(torch.cuda.get_device_name(0)+" "+str(n_gpu))

model_name = 'fifth1.model'
model = pickle.load(open(model_name,'rb'))
model.eval()

data=open("processed_eval.txt")
output=open(sys.argv[1],"w")

print("model_name: ",model_name)
print ("for test prediction")

line_type = 0
tot_lines = 0
qr = True
pr = False
count = 0
n_query = 0
queries = []
paragraphs = []
labels = []
id_pairs = []
query_id = 0
for line in data:
	if qr == True:
	    if line_type%4 == 0:
	        query_id = int(line)
	    elif line_type%4 == 1:
	        query = line.rstrip()
	    elif line_type%4 == 2:
	        n_para = int(line)
	    elif line_type%4 == 3:
	        qr = False
	        pr = True
	        count = 0
	        n_query += 1
	        print ("query no:",n_query)
	        queries = []
	        paragraphs = []
	        labels = []
	        id_pairs = []
	elif pr == True:
		if line_type%4 == 0:
			para_id = int(line)
		elif line_type%4 == 1:
		    para = line.rstrip()
		elif line_type%4 == 2:
		    label = int(line)
		elif line_type%4 == 3:
			count += 1
			queries.append(query)
			paragraphs.append(para)
			labels.append(label)
			id_pairs.append((query_id,para_id))
			if count == n_para:

				qr = True
				pr = False

				tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
				tokenized_queries = [tokenizer.tokenize(query) for query in queries]
				tokenized_paragraph = [tokenizer.tokenize(paragraph) for paragraph in paragraphs]
				segment_mask=[]
				tokenized_texts=[]
				for i in range(0,len(tokenized_queries)):
				    query_seg_mask=[0 for qt in tokenized_queries[i]]
				    para_seg_mask=[1 for pt in tokenized_paragraph[i]]
				    segment_mask.append([0]+query_seg_mask + para_seg_mask)
				    tokenized_texts.append(['[CLS]'] + tokenized_queries[i] + tokenized_paragraph[i] + ['[SEP]'])

				    
				MAX_LEN = 128 #for padding and making input size constant
				input_ids = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_texts]
				input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")
				segment_mask = pad_sequences(segment_mask, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")
				attention_masks = []
				for seq in input_ids:
				    seq_mask = [float(i>0) for i in seq]
				    attention_masks.append(seq_mask)

				test_inputs = input_ids
				test_labels = labels
				test_id_pairs = id_pairs
				test_masks = attention_masks
				test_seg_masks = segment_mask

				test_inputs = torch.tensor(test_inputs)
				test_labels = torch.tensor(test_labels)
				test_id_pairs = torch.tensor(test_id_pairs)
				test_masks = torch.tensor(test_masks)
				test_seg_masks = torch.tensor(test_seg_masks)

				batch_size = 32

				test_data = TensorDataset(test_inputs, test_masks, test_seg_masks, test_labels, test_id_pairs)
				test_sampler = RandomSampler(test_data)
				test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=batch_size)

				eval_accuracy = np.zeros((2,2))
				totpred=0
				totlabel=0
				score={}
				for batch in test_dataloader:
				    batch = tuple(t.to(device) for t in batch)

				    b_input_ids, b_input_mask, b_seg_mask, b_labels, b_id_pairs = batch

				    with torch.no_grad():
				        logits = model(b_input_ids, token_type_ids=b_seg_mask, attention_mask=b_input_mask)

				    logits = logits.detach().cpu().numpy()
				    id_pair_set = b_id_pairs.to('cpu').numpy()

				    for h in range(0,len(b_input_ids)):
				        score[id_pair_set[h][1]]=logits[h][1]

				sorted_score = collections.OrderedDict(sorted(score.items(), key=operator.itemgetter(1), reverse = True))
				rank = 1
				for j in sorted_score:
				    output.write(str(query_id) + " 0 " + str(j) + " " + str(rank) + " " + str(round_nearest(j,0.01)) + " s\n")
				    rank += 1
				tot_lines = tot_lines + len(score)

	line_type += 1
print ("expected number of lines: ",tot_lines)
output.close()