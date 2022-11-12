import pickle
import numpy as np
import pandas as pd
from collections import Counter
from tqdm import tqdm
import time
import ipdb
np.seterr(divide='ignore', invalid='ignore')


def custom_cosine_sim(tokensA, tokensB):
  """
  doc -- (t, tfidf) ...
  query -- (t, tdidf) ...
  """
  # print(tokensA)
  # print(tokensB)
  cosine_sim = 0
  for idA, tfidf in tokensA.items():
    if idA in tokensB:
      cosine_sim += tfidf * tokensB[idA]
  
  return cosine_sim

 
def compute_tf_idf(word_list, op_type, V, N, df):
  """
  Applies the term freq operation on the word list

  Args:
      word_list (token_id, freq): 
      op_type (str): "lan"
      V (int): size of the vocab
      N (int): total no of docs
  Returns:
      final_vector : the vector representing the term
  """
  
  op_type = op_type.lower()
  
  final_dict = dict()
  
  # First operation
  if op_type[0] not in "lan":
    return Exception("Invalid operation")
  
  else:
    for idx, freq in word_list:
      final_dict[idx] = freq
          
    if op_type[0] == "l":
      for dict_idx, dict_freq in word_list:
        final_dict[dict_idx] = np.log(1 + dict_freq)
      
    elif op_type[0] == "a":
      max_val = np.max([freq for token, freq in word_list])
      for dict_idx, dict_freq in word_list:
        final_dict[dict_idx] = 0.5 + 0.5 * dict_freq / (max_val + 1e-20)


  # Second operation
  if op_type[1] not in "ntp":
    return Exception("Invalid operation")
  
  else:
    idf_dict = {idx:1 for idx in final_dict}
    
    if op_type[1] == "t":
      for idx, idf in idf_dict.items():
        idf = np.log(N / df[idx])
      
    elif op_type[1] == "p":
      for idx, idf in idf_dict.items():
        idf = np.log(N / df[idx] - 1)
        if idf < 0:
          idf = 0.0
        
  squares = 0
  for key, val in final_dict.items():
    final_dict[key] = val * idf_dict[key]
    squares += final_dict[key]**2
  squares = np.sqrt(squares)
  
  # Third operation
  if op_type[2] not in "cn":
    return Exception("Invalid operation")
  
  if op_type[2] == "c":
    for idx, val in final_dict.items():
      final_dict[idx] = val / (squares**0.5)
      
  return final_dict


def transpose_inv_idx(inv_idx):
  """
  Takes a transpose of the inverted index

  Args:
      inv_idx (dict): token to (doc_id, freq) mapping

  Returns:
      new_idx(dict): doc_id to (token, freq) mapping
      mapping(dict): token to token_id mapping
      idf(dict): token to idf mapping
  """
  
  # df = dict()
  N = len(inv_idx.keys())
  df = [0]*N
  new_idx = dict()
  mapper = dict()
  
  for idx, key in enumerate(sorted(inv_idx.keys())):
      mapper[key] = idx
      df[idx] = len(inv_idx[key])
      for cord_id, freq in inv_idx[key]:
        if cord_id not in new_idx:
          new_idx[cord_id] = []
        new_idx[cord_id].append((idx, freq))
  
  # df = list(df.values())
  
  return new_idx, mapper, np.array(df)


def get_query_postings(queries, mapper):
  """
  Get a map from query ids to tokens

  Args:
      queries (DataFrame): The queries dataframe
      mapper (dict): token to token_id mapping

  Returns:
      
      query_vector: The mapping from query_id to tokens
  """
  
  query_vector = dict()

  for idx, query in zip(queries["topic-id"], queries["query"]):
    words = query.split()
    query_vector[idx] = []
    tokens = []

    for word in words:
      if word in mapper:
        tokens.append(mapper[word]) 
            
    query_vector[idx] = list(Counter(tokens).items())
  
  return query_vector


def get_ranks(new_idx, query_vector, V, method, output_file, df):
  """Returns a dict containing the query id vs the docs

  Args:
      new_idx (dict): doc_id to (token, freq) mapping
      query_vector (dict): query_id to tokens mapping
      V (int): vocab size
  """
  
  doc_method, query_method = method.split('.')
  start_time = time.time()
  with open(output_file, "w") as f:
    for query_id, query_token_list in tqdm(query_vector.items()):
      scores = []
      f.write(str(query_id) + ",")
      query_vector = compute_tf_idf(query_token_list, query_method, V, len(new_idx.keys()), df)
      # ipdb.set_trace()
      
      cnt=0
      print(f"Query id = {query_id}")
      for doc_id, doc_token_list in new_idx.items():
        cnt+=1
        if cnt%1000==0:
          print(f"processing doc query_id = {query_id}, doc_token_list = {cnt}, time = {time.time()-start_time} sec")
        # ipdb.set_trace()
        doc_vector = compute_tf_idf(doc_token_list, doc_method, V, len(new_idx.keys()), df)
        # ipdb.set_trace()
        scores.append((doc_id, custom_cosine_sim(query_vector, doc_vector)))
        
      scores.sort(key=lambda x: x[1], reverse=True) 
      # ipdb.set_trace()
      scores = scores[:50]
      output = []
      for score in scores:
        output.append(str(score[0]))
        # f.write(str(score[0]) + ",")
      f.write(",".join(output))
      f.write("\n")


if __name__ == "__main__":
  inv_idx_file = "model_queries_10.bin"
  query_file = "./Data/queries_10.txt"
  configs = {
    "lnc.ltc": "Assignment2_10_ranked_list_A.csv",
    "lnc.lpc": "Assignment2_10_ranked_list_B.csv",
    "anc.apc": "Assignment2_10_ranked_list_C.csv"
  }
  
  # Load old inverted index
  with open(inv_idx_file, 'rb') as f:
    inv_idx = pickle.load(f)
  
  # Transpose inverted index for optimizing space
  new_idx, mapper, df = transpose_inv_idx(inv_idx)  
  
  # Get queries to tokens mapping
  queries = pd.read_csv(query_file)
  query_vector = get_query_postings(queries, mapper)
  # ipdb.set_trace()
  
  # Get the ranks and save for different configs
  for config, output_file in configs.items():
    get_ranks(new_idx, query_vector, len(mapper), config, output_file, df)