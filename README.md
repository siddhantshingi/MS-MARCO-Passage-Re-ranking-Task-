# MS MARCO Passage Re-ranking Task
Given a query and top 1000 relevant passages (w.r.t BM25 model) our task is to re-rank the passages to generate a better nDCG metric value.

Dataset is available at cs1160310@hpc.iitd.ac.in/scratch/cse/btech/cs1160310/IR/A2_dataset/

to train this model
`./build.sh
`
`python3 create_train_val_data.py
`
`python3 train.py`

To get complete reranking of the set of queries
`./rerank.sh --query <query_file> --top1000 <top_1000_file> --collection <passage_collection_file>`
