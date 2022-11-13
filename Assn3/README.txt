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

1. For running 3A, run the following command in the terminal:
python Assignment3_10_rocchio.py ./Data model_queries_10.bin ./Data/qrels.csv Assignment2_10_ranked_list_A.csv

Note: We have added the rank list in this folder for convenience
We produce two files:
Assignment3_10_rocchio_PsRF_metrics.csv and Assignment3_10_rocchio_RF_metrics.csv which contains the metrics for the different feedback techniques.

2. For running 3B, run the following command in the terminal:
python Assignment3_10_important_words.py ./Data model_queries_10.bin Assignment2_10_ranked_list_A.csv

Note: We can use different ranked lists for this part. We can add the pseudo relevance ranked list if required.
We produce Assignment3_10_important_words.csv which contains the important words for each query.