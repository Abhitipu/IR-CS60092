model:
	python Assignment2_10_indexer.py ./Data/CORD-19
parse:
	python Assignment2_10_parser.py ./Data/queries.csv

evaluate:
	python Assignment2_10_evaluator.py ./Data/qrels.csv Assignment2_10_ranked_list_$(ch).csv