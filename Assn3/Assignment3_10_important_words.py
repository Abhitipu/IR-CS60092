import pickle
import numpy as np
import pandas as pd
from collections import Counter

def transpose_inv_idx(inv_idx):
  """
  Takes a transpose of the inverted index

  Args:
      inv_idx (dict): token to (doc_id, freq) mapping

  Returns:
      new_idx(dict): doc_id to (token, freq) mapping
      mapping(dict): token to token_id mapping
      df(dict): token to document freq mapping
  """
  
  V = len(inv_idx.keys())
  df = np.zeros(V)
  new_idx = dict()
  mapper = dict()
  
  for idx, key in enumerate(sorted(inv_idx.keys())):
      mapper[key] = idx
      df[idx] = len(inv_idx[key])
      for cord_id, freq in inv_idx[key]:
        if cord_id not in new_idx:
          new_idx[cord_id] = []
        new_idx[cord_id].append((idx, freq))
  
  return new_idx, mapper, df


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
  # print("computing tf id")
  
  op_type = op_type.lower()
  final_vector = np.zeros(V)
  idf_vector = np.ones(V)
  
  # First operation
  if op_type[0] not in "lan":
    return Exception("Invalid operation")
  
  else:
    for idx, freq in word_list:
      final_vector[idx] = freq
          
    if op_type[0] == "l":
      final_vector = np.log(1 + final_vector)
      
    elif op_type[0] == "a":
      final_vector = 0.5 + 0.5 * final_vector / np.max(final_vector)

  # Second operation
  if op_type[1] not in "ntp":
    return Exception("Invalid operation")
  
  else:
    # final_vector = final_vector + 1e-20
    # ipdb.set_trace()
    
    if op_type[1] == "t":
      idf_vector = np.log(N / df)
      
    elif op_type[1] == "p":
      idf_vector = np.log(N / df - 1)
      idf_vector[idf_vector < 0] = 0
  # ipdb.set_trace()
  final_vector = final_vector * idf_vector
  # ipdb.set_trace()
  
  # Third operation
  if op_type[2] not in "cn":
    return Exception("Invalid operation")
  
  return final_vector / (np.linalg.norm(final_vector) + 1e-20) if op_type[2] == "c" else final_vector

if __name__ == "__main__":
    inv_idx_file = "model_queries_10.bin"
    query_file = "./Data/queries_10.txt"
    ranked_file = "./NewRanks_0_ps_relevance.csv"
    output_file = "./Assignment3_10_important_words.csv"
    
    with open(inv_idx_file, 'rb') as f:
        inv_idx = pickle.load(f)

    new_idx, mapper, df = transpose_inv_idx(inv_idx)  
    queries = pd.read_csv(query_file)
    query_vectors = get_query_postings(queries, mapper)
    
    ranked_list_df = pd.read_csv(ranked_file, header=None)
    
    # Vocab size
    V = len(inv_idx)
    N = len(new_idx)
    
    n_docs = 10
    n_words = 5
    inv_mapper = {v: k for k, v in mapper.items()}
    with open(output_file, 'w') as f:
        for index, rows in ranked_list_df.iterrows():
            mean_vec = np.zeros(V)
            for idx, cord_id in enumerate(list(rows[1:n_docs+1])):
                mean_vec = mean_vec + 1 / (idx + 1) * (compute_tf_idf(new_idx[cord_id], "lnc", V, N,df) - mean_vec)
            sorted_indices = np.argsort(mean_vec)[::-1]
            f.write(f"{rows[0]}:{','.join([inv_mapper[idx] for idx in sorted_indices[:n_words]])}\n")        
