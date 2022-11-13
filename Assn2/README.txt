Dependencies


----------------------------------------------------------------------------------------------------
Running Instructions:

Prerequisites:

1. We have slightly modified the indexer file. We have added the indexer file for convenience. 

To construct the inverted index use:
python Assignment2_10_indexer.py ./Data/CORD-19

This saves the output in model_queries_10.bin. Now we use this file for the following parts.

We also assume that the path given contains the file queries_10.txt. 
We have added the file in the Data folder for convenience.

Steps:

1. For running 2A, run the following command in the terminal:
python Assignment2_10_ranker.py ./Data model_queries_10.bin


2. For running 2B, run the following commands in the terminal:
python Assignment2_10_evaluator.py ./Data/qrels.csv Assignment2_10_ranked_list_A.csv
python Assignment2_10_evaluator.py ./Data/qrels.csv Assignment2_10_ranked_list_B.csv
python Assignment2_10_evaluator.py ./Data/qrels.csv Assignment2_10_ranked_list_C.csv

These correspond to the different tf-idf schemes. The results will be stored in 
Asssignment2_10_metrics_A.csv, Assignment2_10_metrics_B.csv, Assignment2_10_metrics_C.csv respectively.

