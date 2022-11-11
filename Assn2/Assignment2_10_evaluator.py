import pickle
import numpy as np
import pandas as pd
from collections import Counter
import sys
import ipdb

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

def mean_average_precision(ground_truth, rank_list, k):
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

    
if __name__ == "__main__":
  n = len(sys.argv)
  # print(n)
  if n < 3:
    raise Exception("Error in format")
  
  ground_truth_file = sys.argv[1]
  ranked_file = sys.argv[2]
  output_file = "./Assignment2_10_metrics_"+ranked_file[-5:]
  # print(output_file)
  # exit()
  ground_truth_df = pd.read_csv(ground_truth_file)
  ranked_list_df = pd.read_csv(ranked_file, header=None)
  ranked_list = [["dummy"]]
  for index, rows in ranked_list_df.iterrows():
      # Create list for the current row
      # omitting the query number
      ranked_list.append(list(rows[1:]))

  # ranked_list[query_id] -> order of retrieval (list of cord_ids)    

  # list of dictionaries, each mapping cord_id -> relevance score, index in list = query_id
  ground_truth = get_rel_dict(ground_truth_df)
  
  MAP10 = mean_average_precision(ground_truth, ranked_list, 20)
  MAP20 = mean_average_precision(ground_truth, ranked_list, 20)

  NDCG10 = ndcg(ground_truth, ranked_list, 10)
  NDCG20 = ndcg(ground_truth, ranked_list, 20)

  data_dict = {
    'query_id' : [qid for qid in range(1, 36)] + ["Avg"],
    'MAP@10' : MAP10 + [np.mean(MAP10)],
    'MAP@20' : MAP20 + [np.mean(MAP20)],
    'NDCG@10' : NDCG10 + [np.mean(NDCG10)],
    'NDCG@20' : NDCG20 + [np.mean(NDCG20)],
    
  }
  result_df = pd.DataFrame(data_dict)
  result_df.to_csv(output_file, index=False)

