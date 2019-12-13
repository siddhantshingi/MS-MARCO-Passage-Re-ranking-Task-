import sys
import pandas as pd

collection_file = sys.argv[3]
query_file = sys.argv[1]
top1000_file = sys.argv[2]
print (query_file, top1000_file, collection_file)

pardata=pd.read_csv(collection_file, delimiter='\t', header=None, names=['par_id', 'par_text'])
print("pardata loaded")
pardict=dict(zip(pardata.par_id.values,pardata.par_text.values))
print("predict created")

querydata = pd.read_csv(query_file, delimiter='\t', header=None, names=['query_id', 'query_text'])
print("querydata loaded")
querydict = dict(zip(querydata.query_id.values,querydata.query_text.values))
print("querydict created")

data_top1000 = open(top1000_file)
Dict = {}
for cnt, line in enumerate(data_top1000):
    line_list = line.split()
    para_id = int(line_list[1])
    query_id = int(line_list[0])
    if query_id in Dict:
        Dict[query_id].append(para_id)
    else:
        Dict[query_id] = [para_id]
print ("Top1000 loaded")
print ("Top1000_Dict loaded")

data_output = open("processed_eval.txt","w")
data_stat = open("processed_eval.stat.txt","w")
for query in Dict:
    data_output.write(str(query) + "\n")
    data_output.write(querydict[query] + "\n")
    data_output.write(str(len(Dict[query])) + "\n")
    data_output.write("\n")
    for para in Dict[query]:
        data_output.write(str(para) + "\n")
        data_output.write(pardict[para] + "\n")
        data_output.write("0\n")
        data_output.write("\n")
    data_stat.write(str(query) + " " + str(len(Dict[query])) + "\n")