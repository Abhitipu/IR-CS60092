Dependencies

backcall==0.2.0
click==8.1.3
colorama==0.4.5
decorator==5.1.1
importlib-metadata==4.12.0
ipdb==0.13.9
ipython==7.34.0
jedi==0.18.1
joblib==1.1.0
matplotlib-inline==0.1.6
nltk==3.7
numpy==1.21.6
pandas==1.3.5
parso==0.8.3
pickleshare==0.7.5
prompt-toolkit==3.0.32
Pygments==2.13.0
python-dateutil==2.8.2
pytz==2022.2.1
regex==2022.8.17
six==1.16.0
toml==0.10.2
tqdm==4.64.1
traitlets==5.5.0
typing_extensions==4.3.0
wcwidth==0.2.5
zipp==3.8.1

----------------------------------------------------------------------------------------------------
Running Instructions:

Prerequisites:

1. We have slightly modified the indexer file. We have added the indexer file for convenience. 

To construct the inverted index use:
python Assignment2_10_indexer.py ./Data/CORD-19

This saves the output in model_queries_10.bin. Now we use this file for the following parts.

Steps:

1. For running 2A, run the following command in the terminal:
python Assignment2_10_ranker.py ./Data model_queries_10.bin

Note: Here we assume that the path given contains the file queries_10.txt. 
We have added the file in the Data folder for convenience.

2. For running 2B, run the following commands in the terminal:
python Assignment2_10_evaluator.py ./Data/qrels.csv Assignment2_10_ranked_list_A.csv
python Assignment2_10_evaluator.py ./Data/qrels.csv Assignment2_10_ranked_list_B.csv
python Assignment2_10_evaluator.py ./Data/qrels.csv Assignment2_10_ranked_list_C.csv

These correspond to the different tf-idf schemes. The results will be stored in 
Asssignment2_10_metrics_A.csv, Assignment2_10_metrics_B.csv, Assignment2_10_metrics_C.csv respectively.

