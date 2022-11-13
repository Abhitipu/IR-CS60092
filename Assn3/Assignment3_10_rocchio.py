import pickle
import numpy as np
import pandas as pd
from collections import Counter
from tqdm import tqdm
import time
import ipdb
import sys

def custom_cosine_sim(tokensA, tokensB):
  """
  doc -- (t, tfidf) ...
  query -- (t, tdidf) ...
  return cosine similarity
  """
  cosine_sim = 0
  for idA, tfidf in tokensA.items():
    if idA in tokensB:
      cosine_sim += tfidf * tokensB[idA]
  
  return cosine_sim

def custom_add(tokensA, tokensB):
  """
  doc -- (t, tfidf) ...
  query -- (t, tdidf) ...
  returns sum of two vectors
  """
  term_ids = set()

  for term_id in tokensA:
    term_ids.add(term_id)

  for term_id in tokensB:
    term_ids.add(term_id)

 
  sumToken = dict()
  for term_id in term_ids:
    sumToken[term_id] = 0
    if term_id in tokensA:
      sumToken[term_id] += tokensA[term_id]
    if term_id in tokensB:
      sumToken[term_id] += tokensB[term_id]
  return sumToken

def compute_average_precision(ground_truth, rank_list, k):
  """
    Ground truth : query_id -> cord_id -> (judgement, iteration)
    rank list: query_id -> ranked list of cord_id
    Returns average precision for each query
  """
  # non zero judgments are relevant
  query_id = 1
  average_precision = []
  for row in rank_list[1:]:
    prec = [] #prec@k
    total_cords = 0
    relevant_cords = 0
    loop_k = min(k, len(row))
    for i in range(loop_k):
      total_cords += 1
      cord_id = row[i]
      if cord_id in ground_truth[query_id]:
        if ground_truth[query_id][cord_id][0]!=0:
          relevant_cords += 1
          prec.append(relevant_cords/(total_cords))
    if(len(prec) == 0):
      average_precision.append(0)
    else:
      average_precision.append(np.mean(prec))
    # ipdb.set_trace()
    query_id+=1
  return average_precision  #note the offset

def ndcg(ground_truth, rank_list, k):
  """
    Computes normalized discounted cumulative gain
    for each query with k as the cutoff
  """
  query_id = 1
  ndcg = []
  for row in rank_list[1:]:
    # dcg = []
    cur_dcg = 0
    position = 0
    loop_k = min(k, len(row))
    values = []
    for i in range(loop_k):
      cord_id = row[i]
      position += 1
      val = 0
      discount = 1
      if position > 1:
        discount = np.log2(position)
      if cord_id in ground_truth[query_id]:
        val = ground_truth[query_id][cord_id][0]
      values.append(val)
      cur_dcg+= val/discount

    ideal_ar = []
    for key, val in ground_truth[query_id].items():
      ideal_ar.append(val[0])

    ideal_ar.sort(reverse = True)
    loop_k = min(k, len(ideal_ar))
    i_dcg=0
    for i in range(loop_k):
      discount=1
      if i>0:
        discount = np.log2(i+1)
      i_dcg+=ideal_ar[i]/discount
    # ipdb.set_trace()
    ndcg.append(cur_dcg/i_dcg)
    query_id+=1

  return ndcg

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

def get_rel_dict(ground_truth_df):
  ground_truth = [{} for i in range(36)]
  # queryid -> {} cord-id-> (judgement, iteration)
  for index, rows in ground_truth_df.iterrows():
    query_id = rows['topic-id']
    if rows['cord-id'] in ground_truth[query_id]:
      if rows['iteration']>ground_truth[query_id][rows['cord-id']][1]:
        ground_truth[query_id][rows['cord-id']] = (rows['judgement'], rows['iteration'])
    else:
      ground_truth[query_id][rows['cord-id']] = (rows['judgement'], rows['iteration'])
        
  return ground_truth

def compute_tf_idf(word_list, op_type, V, N, df):
  """
  Applies the term freq operation on the word list

  Args:
      word_list (token_id, freq): 
      op_type (str): "lan"
      V (int): size of the vocab
      N (int): total no of docs
  Returns:
      final_dict : the vector representing the (t, f) pairs
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
        idf_dict[idx] = np.log(N / df[idx])
      
    elif op_type[1] == "p":
      for idx, idf in idf_dict.items():
        idf_dict[idx] = np.log(N / df[idx] - 1)
        if idf_dict[idx] < 0:
          idf_dict[idx] = 0.0
        
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
      final_dict[idx] = val / (squares + 1e-20)
  # ipdb.set_trace()    
  return final_dict

# (word_list, op_type, V, N, df):
def relevance_feedback(ground_truth, ranked_list, doc_tf_idf, k):
  """
    Returns an array of tuples where each entry is (positive_feedback, negative_feedback)
    positive feedback = (1 / |D_R|) * sum(d_R)
    negative feedback = (1 / |D_NR|) * sum(d_NR)
    One entry for each query
  """
  # judgments == 2 are relevant
  
  query_id = 1

  feedback = []
  
  for row in ranked_list[1:]:
    # pos_feedback = np.zeros(V)
    pos_feedback = dict()
    pos_vector_count = 0
    
    neg_feedback = dict()
    neg_vector_count = 0

    loop_k = min(k, len(row))
    for i in range(loop_k):
      cord_id = row[i]
      doc_vector = doc_tf_idf[cord_id]
      if cord_id in ground_truth[query_id] and (ground_truth[query_id][cord_id][0]==2):
        # relevant doc
        pos_feedback = custom_add(pos_feedback, doc_vector)
        pos_vector_count+=1
      else:
        # irrelevant doc
        neg_feedback = custom_add(neg_feedback, doc_vector)
        neg_vector_count+=1
    if(pos_vector_count != 0):
      for idx, v in pos_feedback.items():
        pos_feedback[idx] /= pos_vector_count
    
    if(neg_vector_count != 0):
      for idx, v in neg_feedback.items():
        neg_feedback[idx] /= neg_vector_count
    
    query_id +=1
    feedback.append((pos_feedback, neg_feedback))
    
  return feedback

def pseudo_relevance_feedback(ground_truth, ranked_list, doc_tf_idf, k):
  """
    Returns an array of tuples where each entry is (positive_feedback, negative_feedback)
    Positive feedback = (1 / |D_R|) * sum(d_R)
    Negative feedback = NULL
    One entry for each query
  """
  # judgments == 2 are relevant
  
  query_id = 1

  feedback = []
  
  for row in ranked_list[1:]:
    pos_feedback = dict()
    pos_vector_count = 0
    
    neg_feedback = dict()
    neg_vector_count = 0

    loop_k = min(k, len(row))
    for i in range(loop_k):
      cord_id = row[i]
      doc_vector = doc_tf_idf[cord_id]
      
      
      pos_feedback = custom_add(pos_feedback, doc_vector)
      pos_vector_count+=1
      
    if(pos_vector_count != 0):
      for idx, v in pos_feedback.items():
        pos_feedback[idx] /= pos_vector_count

    query_id +=1
    feedback.append((pos_feedback, neg_feedback))
    
  return feedback

def modify_query(query_vector, feedback, config):
  """
    Returns the modified query vector
    Modified query = (1 - alpha) * query_vector + alpha * (pos_feedback - beta * neg_feedback)
  """
  alpha = config[0]
  beta = config[1]
  gamma = config[2]
  pos_feedback = feedback[0]
  neg_feedback = feedback[1]
  term_ids = set()

  for term_id in query_vector:
    term_ids.add(term_id)

  for term_id in pos_feedback:
    term_ids.add(term_id)

  for term_id in neg_feedback:
    term_ids.add(term_id)

  modified_query = dict()
  for term_id in term_ids:
    modified_query[term_id] = 0
    if term_id in query_vector:
      modified_query[term_id] += alpha * query_vector[term_id]
    if term_id in pos_feedback:
      modified_query[term_id] += beta * pos_feedback[term_id]
    if term_id in neg_feedback:
      modified_query[term_id] -= gamma * neg_feedback[term_id]
  return modified_query


def get_ranks(query_tf_idf, doc_tf_idf, output_file):
  """Returns a dict containing the query id vs the docs

  Args:
      new_idx (dict): doc_id to (token, freq) mapping
      query_vector (dict): query_id to tokens mapping
      V (int): vocab size
  """
  
  ranked_list = []
  with open(output_file, "w") as f:
    for query_id, query_vector in tqdm(query_tf_idf.items()):
      scores = []
      f.write(str(query_id) + ",")
      
      cnt=0
      for doc_id, doc_vector in doc_tf_idf.items():
        cnt+=1
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
      ranked_list.append(output)
  return ranked_list  
      
if __name__ == "__main__":
  n = len(sys.argv)
  if n < 5:
    print("Error format")
    
  #filenames
  query_file = f"{sys.argv[1]}/queries_10.txt"
  inv_idx_file = sys.argv[2]
  ground_truth_file = sys.argv[3]
  ranked_file = sys.argv[4]
  
  relevance_file = "./Assignment3_10_rocchio_RF_metrics.csv"
  pseudo_relevance_file = "./Assignment3_10_rocchio_PsRF_metrics.csv"

  #ground truth list of dict
  ground_truth_df = pd.read_csv(ground_truth_file)
  ground_truth = get_rel_dict(ground_truth_df)

  ranked_list_df = pd.read_csv(ranked_file, header=None)
  ranked_list = [["dummy"]]
  for index, rows in ranked_list_df.iterrows():
      # Create list for the current row
      # omitting the query number
      ranked_list.append(list(rows[1:]))
      
  #index
  method = "lnc.ltc"
  doc_method, query_method = method.split('.')

  with open(inv_idx_file, 'rb') as f:
    inv_idx = pickle.load(f)

  new_idx, mapper, df = transpose_inv_idx(inv_idx)  

  #queries
  queries = pd.read_csv(query_file)
  query_vectors = get_query_postings(queries, mapper)

  
  V = len(mapper)
  N = len(new_idx)
  
  doc_tf_idf = {doc_id: compute_tf_idf(doc_token_list, doc_method, V, N, df) for doc_id, doc_token_list in new_idx.items()}
  feedback = relevance_feedback(ground_truth, ranked_list, doc_tf_idf, 20)# 2 vector
  pseudo_feedback = pseudo_relevance_feedback(ground_truth, ranked_list, doc_tf_idf, 10)
  query_tf_idf = {query_id: compute_tf_idf(query_token_list, query_method, V, N, df) for query_id, query_token_list in query_vectors.items()}  

  # alpha, beta, gamma
  rf_config = [
    [1, 1, 0.5], 
    [0.5, 0.5, 0.5], 
    [1, 0.5, 0]
  ]

  ranked_lists= []
  for idx, config in enumerate(rf_config):
    # one of six
    modified_query_tf_idf = {query_id: modify_query(query_tf_idf[query_id],feedback[query_id-1], config ) for query_id, query_token_list in query_vectors.items()}  
    ranked_list = get_ranks(modified_query_tf_idf, doc_tf_idf, f"NewRanks_{idx}_relevance.csv")
    
    ranked_lists.append([["dummy"]] + ranked_list) # for fixing the offset
    modified_query_tf_idf = {query_id: modify_query(query_tf_idf[query_id],pseudo_feedback[query_id-1], config ) for query_id, query_token_list in query_vectors.items()}  
    ranked_list = get_ranks(modified_query_tf_idf, doc_tf_idf, f"NewRanks_{idx}_ps_relevance.csv")
    ranked_lists.append(ranked_list)
    

  # Merging the six results into two
  ctr = 0 
  with open(pseudo_relevance_file, "w") as f:
    f.write("alpha, beta, gamma, mAP@20, NDCG20 \n")
    pass

  with open(relevance_file, "w") as f:
    f.write("alpha, beta, gamma, mAP@20, NDCG20 \n")
    pass

  for ranked_list in ranked_lists:
    fname = ""
    mAP20 = np.mean(compute_average_precision(ground_truth, ranked_list, 20))
    nDCG = np.mean(ndcg(ground_truth, ranked_list, 20))
    k = ctr
    if ctr%2 == 1:
      k = k - 1
    k = k >> 1
    
    if ctr%2 == 0:
      fname = relevance_file
    else:
      fname = pseudo_relevance_file 
    
    with open(fname, "a") as f:
      f.write(str(rf_config[k][0])+", ")
      f.write(str(rf_config[k][1])+", ")
      f.write(str(rf_config[k][2])+", ")
      f.write(str(mAP20)+", ")
      f.write(str(nDCG)+"\n") 
          
    ctr = ctr + 1 ; 