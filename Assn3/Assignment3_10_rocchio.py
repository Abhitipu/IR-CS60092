import pickle
import numpy as np
import pandas as pd
from collections import Counter
from tqdm import tqdm
import time
import ipdb
import threading
general_lock = threading.Lock()

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

# (word_list, op_type, V, N, df):
def relevance_feedback(ground_truth, ranked_list, new_idx, V, method, df, k):
  # judgments == 2 are relevant
  doc_method, query_method = method.split('.')
  
  query_id = 1

  feedback = []
  
  for row in ranked_list[1:]:
    pos_feedback = np.zeros(V)
    pos_vector_count = 0
    
    neg_feedback = np.zeros(V)
    neg_vector_count = 0
    # prec = [] #prec@k
    # relevant_cords = 0

    loop_k = min(k, len(row))
    # ipdb.set_trace()
    for i in range(loop_k):
      cord_id = row[i]
      doc_vector = compute_tf_idf(new_idx[cord_id], doc_method, V, len(new_idx.keys()), df)
      if cord_id in ground_truth[query_id] and (ground_truth[query_id][cord_id][0]==2):
        # relevant doc
        # get tf idf vector of the doc, add vector to pos_feedback, increment pos vector count 
        # relevant_cords += 1
        # if(pos_feedback is None):
        #   pos_feedback = doc_vector
        # else:
        pos_feedback += doc_vector
        pos_vector_count+=1
      else:
        # irrelevant doc
        # get tf idf vector of the doc, add vector to neg_feedback, increment neg vector count 
        # if(neg_feedback is None):
        #   neg_feedback = doc_vector
        # else:
        neg_feedback += doc_vector
        neg_vector_count+=1
    # print(query_id, len(prec), k)
    # 
    if(pos_vector_count != 0):
      pos_feedback/=pos_vector_count
    if(neg_vector_count != 0):
      neg_vector_count/=neg_vector_count
    query_id +=1
    feedback.append((pos_feedback, neg_feedback))
    
  return feedback

def pseudo_relevance_feedback(ground_truth, ranked_list, new_idx, V, method, df, k):
  # judgments == 2 are relevant
  doc_method, query_method = method.split('.')
  
  query_id = 1

  feedback = []
  
  for row in ranked_list[1:]:
    pos_feedback = np.zeros(V)
    pos_vector_count = 0
    
    neg_feedback = np.zeros(V)
    neg_vector_count = 0
    # prec = [] #prec@k
    # relevant_cords = 0

    loop_k = min(k, len(row))
    # ipdb.set_trace()
    for i in range(loop_k):
      cord_id = row[i]
      doc_vector = compute_tf_idf(new_idx[cord_id], doc_method, V, len(new_idx.keys()), df)
      
      pos_feedback += doc_vector
      pos_vector_count+=1
      
    # print(query_id, len(prec), k)
    # 
    if(pos_vector_count != 0):
      pos_feedback/=pos_vector_count

    query_id +=1
    feedback.append((pos_feedback, neg_feedback))
    
  return feedback
def modify_query(query_vector, feedback, config):
  alpha = config[0]
  beta = config[1]
  gamma = config[2]
  pos_feedback = feedback[0]
  neg_feedback = feedback[1]
  modified_query = alpha * query_vector + beta*pos_feedback - gamma*neg_feedback
  return modified_query

def thread_target(word_list, op_type, V, N, df, query_vectors, doc_id, scores):
    doc_vector = compute_tf_idf(word_list, op_type, V, N, df)
    values = []
    for q_vec in query_vectors:
      values.append(np.dot(q_vec, doc_vector))
    # value = np.dot(query_vector, doc_vector)
    # temp_scores = []
    # for val in values:
    #   temp_scores.append((doc_id, val))

    idx = 0
    general_lock.acquire()
    for val in values:
      scores[idx].append((doc_id, val))
      idx+=1
    # scores.append(temp_scores) # temp_scores will be 6 different score
    general_lock.release()

def get_ranks(new_idx, query_vectors, V, method, output_file, df):
  # ipdb> len(modified_queries)
  # 35
  # ipdb> len(modified_queries[0])
  # 6
  # ipdb> len(modified_queries[0][0])
  # 146080 
  """Returns a dict containing the query id vs the docs

  Args:
      new_idx (dict): doc_id to (token, freq) mapping
      query_vector (dict): query_id to tokens mapping
      V (int): vocab size
  """
  
  doc_method, query_method = method.split('.')
  start_time = time.time()
  for i in range(len(query_vectors[0])):
    with open(output_file+str(i)+".csv", "w") as f:
      pass
    
  query_id = 1
  for query_vector in tqdm(query_vectors):
    scores = [[] for i in range(len(query_vector))]
    
    
    # 6 * 50000
    outputs = [[str(query_id)] for i in range(len(query_vector))]

    # ipdb.set_trace()
    
    cnt=0
    print(f"Query id = {query_id}")
    threads = []
    lenValue = len(new_idx.keys())
    for doc_id, doc_token_list in new_idx.items():
      cnt+=1
      if cnt%1000==0:
        for t in threads:
          t.join()
        threads = []
        print(f"processing doc query_id = {query_id}, doc_token_list = {cnt}, time = {time.time()-start_time} sec")
      if cnt%50 == 0:
        for t in threads:
          t.join()
        threads = []
      # ipdb.set_trace()
      # doc_vector = compute_tf_idf(doc_token_list, doc_method, V, len(new_idx.keys()), df)
      # # ipdb.set_trace()
      # scores.append((doc_id, np.dot(query_vector, doc_vector)))
      t = threading.Thread(target=thread_target, args=(doc_token_list, doc_method, V, lenValue, df, query_vector, doc_id, scores))
      t.start()
      threads.append(t)

    for t in threads:
      t.join()
    
    for i in range(len(scores)):
      scores[i].sort(key=lambda x: x[1], reverse=True)
      scores[i] = scores[i][:50]
      for score in scores[i]:
        outputs[i].append(str(score[0]))
      with open(output_file+str(i)+".csv", "a") as f:  
        f.write(",".join(outputs[i]))
        f.write("\n")
    query_id+=1
    # scores.sort(key=lambda x: x[1], reverse=True) 
    # ipdb.set_trace()
    # scores = scores[:50]
    
    # for score in scores:
    #   output.append(str(score[0]))
    #   # f.write(str(score[0]) + ",")
    # with open(output_file, "a") as f:  
    #   f.write(",".join(output))
    #   f.write("\n")
      
if __name__ == "__main__":
  #filenames
  inv_idx_file = "model_queries_10.bin"
  query_file = "./Data/queries_10.txt"
  ground_truth_file = "./Data/qrels.csv"
  ranked_file = "./Assignment2_10_ranked_list_A.csv"

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

  with open(inv_idx_file, 'rb') as f:
    inv_idx = pickle.load(f)

  new_idx, mapper, df = transpose_inv_idx(inv_idx)  

  #queries
  queries = pd.read_csv(query_file)
  query_vectors = get_query_postings(queries, mapper)

  feedback = relevance_feedback(ground_truth, ranked_list, new_idx, len(mapper), method, df, 20)# 2 vector
  pseudo_feedback = pseudo_relevance_feedback(ground_truth, ranked_list, new_idx, len(mapper), method, df, 10)

  # ipdb.set_trace()

  rf_config = [
    [1, 1, 0.5], 
    [0.5, 0.5, 0.5], 
    [1, 0.5, 0]
  ]

  
  vocab_size = len(mapper)
  total_docs = len(new_idx.keys())

  modified_queries = []
  
  for query_id, query_token_list in tqdm(query_vectors.items()):
      query_vector = compute_tf_idf(query_token_list, "ltc", vocab_size, total_docs, df)
      modified_query = []
      for config in rf_config:
        modified_query.append(modify_query(query_vector, feedback[query_id-1], config))
        modified_query.append(modify_query(query_vector, pseudo_feedback[query_id-1], config))
      modified_queries.append(modified_query)

  print(modified_query)
  # ipdb.set_trace()    
  get_ranks(new_idx, modified_queries, vocab_size, method, "NewRanks_", df)
    