import pandas as pd
import pickle
import random

# pardata=pd.read_csv("/scratch/cse/btech/cs1160323/data-release-part/collection.tsv", delimiter='\t', header=None, names=['par_id', 'par_text'])
# print("pardata loaded")
# pardict=dict(zip(pardata.par_id.values,pardata.par_text.values))
# print("pardict created")
# print ("len of pardict: ",len(pardict))
dbfile = open('./pickle_files/para_collection', 'rb')      
pardict = pickle.load(dbfile) 
print("paradata loaded")
dbfile.close() 
print ("pardict created")
print ("len of pardict: ",len(pardict))

# querydata = pd.read_csv("/scratch/cse/btech/cs1160323/data-release-part/queries.train_part.tsv", delimiter='\t', header=None, names=['query_id', 'query_text'])
# print("querydata loaded")
# querydict = dict(zip(querydata.query_id.values,querydata.query_text.values))
# print("querydict created")
# print ("len of querydict: ",len(querydict))
dbfile = open('./pickle_files/query_train', 'rb')      
querydict = pickle.load(dbfile) 
print("querydata loaded")
dbfile.close() 
print ("querydict created")
print ("len of querydict: ",len(querydict))


# qrelsdata = pd.read_csv("/scratch/cse/btech/cs1160323/data-release-part/qrels.train.tsv", delimiter='\t', header=None, names=['query_id', 'random_shit1', 'para_id', 'random_shit2'])
# print("qrels loaded")
# query_ids = qrelsdata.query_id.values.tolist()
# para_ids = qrelsdata.para_id.values.tolist()
# Dict_qrels = {}
# for i in range(0,len(query_ids)):
# 	query_id = int(query_ids[i])
# 	para_id = int(para_ids[i])
# 	if query_id in Dict_qrels:
# 		Dict_qrels[query_id].append(para_id)
# 	else:
# 		Dict_qrels[query_id] = [para_id]
dbfile = open('./pickle_files/qrel_train', 'rb')      
Dict_qrels = pickle.load(dbfile) 
dbfile.close() 
print ("qrels_Dict created")
print ("len of Dict_qrels: ",len(Dict_qrels))

# data_input_top1000 = open("/scratch/cse/btech/cs1160323/data-release-part/top1000.train_part.txt",'r')
# Dict_top1000 = {}
# for cnt, line in enumerate(data_input_top1000):
# 	line_list = line.split()
# 	para_id = int(line_list[1])
# 	query_id = int(line_list[0])
# 	if query_id in Dict_top1000:
# 		Dict_top1000[query_id].append(para_id)
# 	else:
# 		Dict_top1000[query_id] = [para_id]
dbfile = open('./pickle_files/top1000_train', 'rb')      
Dict_top1000 = pickle.load(dbfile) 
dbfile.close() 
print ("Top1000 loaded")
print ("Top1000_Dict created")
print ("len of Dict_top1000: ",len(Dict_top1000))

tot_queries = len(Dict_top1000)
n_train = int(0.9*tot_queries)
n_val = tot_queries - n_train
print ("train data len: ",n_train)
print ("val data len: ",n_val)

data_output_train = open("train_data_2/data.txt","w")
data_train_stat = open("train_data_2/stat.txt","w")
# data_val = open("val_data_2/data.txt","w")
# data_val_stat = open("val_data_2/stat.txt","w")
count = 0
for query in Dict_top1000:
	if (count < n_train):
		if (query not in Dict_qrels):
			print ("query in Dict_top1000 not in Dict_qrels")
			count += 1
			continue
		rel_list = []
		for x in Dict_qrels[query]:
			rel_list.append(x)
		tot_list = Dict_top1000[query]
		for para in Dict_qrels[query]:
			data_output_train.write(querydict[query] + "\n")
			data_output_train.write(pardict[para] + "\n")
			data_output_train.write("1\n")
			data_output_train.write("\n")
			data_output_train.write(querydict[query] + "\n")
			data_output_train.write(pardict[para] + "\n")
			data_output_train.write("1\n")
			data_output_train.write("\n")

			non_rel_list = [value for value in tot_list if value not in rel_list]
			non_rel_para = non_rel_list[random.randint(0, len(non_rel_list)-1)]
			data_output_train.write(querydict[query] + "\n")
			data_output_train.write(pardict[non_rel_para] + "\n")
			data_output_train.write("0\n")
			data_output_train.write("\n")
			rel_list.append(non_rel_para)

		data_train_stat.write(str(query) + " " + str(len(rel_list)*2) + "\n")
	elif (count >= n_train):
		if (query not in querydict):
			print ("query not found in querydict")
			count += 1
			continue
		data_val.write(str(query) + "\n")
		data_val.write(querydict[query] + "\n")
		data_val.write(str(len(Dict_top1000[query])) + "\n")
		data_val.write("\n")
		for para in Dict_top1000[query]:
			data_val.write(str(query) + " " + str(para) + "\n")
			data_val.write(pardict[para] + "\n")
			data_val.write("0\n")
			data_val.write("\n")
		data_val_stat.write(str(query) + " " + str(len(Dict_top1000[query])) + "\n")
	print ("loop count: ",count)
	count += 1	