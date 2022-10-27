import pickle
import numpy as np
import pandas as pd
from collections import Counter

def get_doc_frequency(d, idx):


    pass

def get_query_frequency(q, idx):


    pass

if __name__ == "__main__":
    filename = "model_queries_10.bin"
   
    with open(filename, 'rb') as f:
        inv_idx = pickle.load(f)
   
    # Get idf from length of posting list
    idf = dict()
    N = len(inv_idx.keys())
    new_idx = dict()
    mapper = dict()
    # inverting inv_idx
    for idx, key in enumerate(sorted(inv_idx.keys())):
        mapper[key] = idx
        idf[key] = np.log(N / len(inv_idx[key]))
        for cord_id, freq in inv_idx[key]:
          if cord_id not in new_idx:
            new_idx[cord_id] = []
          new_idx[cord_id].append((idx, freq))
          # doc -- (w1, f1), (w2, f2) ..
   
   
    query_file = "./Data/queries_10.txt"
    queries = pd.read_csv(query_file)
    query_vector = dict()


    for idx, query in zip(queries["topic-id"], queries["query"]):
        w = query.split()
        query_vector[idx] = []
        ls = []

        for word in w :
            if word in mapper:
                ls.append(mapper[word]) 

            else: 
                # maa chudaye
                pass

        query_vector[idx] = list(dict(Counter(ls)).items())

        
    # word -- id
    # for query in queries:

