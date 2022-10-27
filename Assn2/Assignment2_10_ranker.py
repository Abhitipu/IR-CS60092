import pickle
import numpy as np
import pandas as pd
from collections import Counter


def compute_tf_idf(word_list, op_type, V, N):
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
  final_vector = np.zeros(V)
  
  # First operation
  if op_type[0] not in "lan":
    return Exception("Invalid operation")
  
  else:
    for idx, freq in word_list:
      final_vector[idx] = freq
          
    if op_type == "l":
      final_vector = np.log(1 + final_vector)
      
    elif op_type == "a":
      final_vector = 0.5 + 0.5 * final_vector / np.max(final_vector)

  # Second operation
  if op_type[1] not in "ntp":
    return Exception("Invalid operation")
  
  else:
    if op_type == "t":
      final_vector = np.log(N / final_vector)
      
    elif op_type == "p":
      final_vector = np.log(N / final_vector - 1)
  
  # Third operation
  if op_type[2] not in "cn":
    return Exception("Invalid operation")
  
  return final_vector / np.linalg.norm(final_vector) if op_type == "c" else final_vector


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
  
  idf = dict()
  N = len(inv_idx.keys())
  new_idx = dict()
  mapper = dict()
  
  for idx, key in enumerate(sorted(inv_idx.keys())):
      mapper[key] = idx
      idf[key] = np.log(N / len(inv_idx[key]))
      for cord_id, freq in inv_idx[key]:
        if cord_id not in new_idx:
          new_idx[cord_id] = []
        new_idx[cord_id].append((idx, freq))
  
  return new_idx, mapper, idf


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


def get_ranks(new_idx, query_vector, V, method, output_file):
  """Returns a dict containing the query id vs the docs

  Args:
      new_idx (dict): doc_id to (token, freq) mapping
      query_vector (dict): query_id to tokens mapping
      V (int): vocab size
  """
  
  doc_method, query_method = method.split('.')
  
  with open(output_file, "w") as f:
    for query_id, query_token_list in query_vector.items():
      scores = []
      f.write(str(query_id) + ",")
      query_vector = compute_tf_idf(query_token_list, query_method, V, len(new_idx.keys()))
      
      for doc_id, doc_token_list in doc_vector.items():
        doc_vector = compute_tf_idf(doc_token_list, doc_method, V, len(new_idx.keys()))
        scores.append((doc_id, np.dot(query_vector, doc_vector)))
        
      scores.sort(key=lambda x: x[1], reverse=True) 
      scores = scores[:50]
      for score in scores:
        f.write(str(score[0]) + ",")
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
  new_idx, mapper, idf = transpose_inv_idx(inv_idx)  
  
  # Get queries to tokens mapping
  queries = pd.read_csv(query_file)
  query_vector = get_query_postings(queries, mapper)
  
  # Get the ranks and save for different configs
  for config, output_file in configs.items():
    get_ranks(new_idx, query_vector, len(mapper), config, output_file)