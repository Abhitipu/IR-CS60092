import pickle
import pandas as pd

import sys

def mergePostingList(A, B):
    """
        Returns a list of intersection of two postings list A and B
    """
    len_a, len_b = len(A), len(B)
    i , j = 0, 0
    result = []
    while i < len_a and j < len_b:
        if A[i] < B[j]:
            i += 1
        elif A[i] == B[j]:
            result.append(A[i])
            i += 1
            j += 1
        else:
            j += 1
    return result

def retrieveDocuments(inverted_idx, query):
    """
        returns the list of cord_ids 
        corresponding to the documents 
        containing the query tokens.
    """
    if(query[0] not in inverted_idx.keys()):
        return []
    
    base_postingList = inverted_idx[query[0]]

    for i in range(1, len(query)):
        if (query[i] not in inverted_idx.keys()):
            return []
        base_postingList = mergePostingList(base_postingList, inverted_idx[query[i]])
    
    return base_postingList

if __name__ == "__main__":
    n = len(sys.argv)
    
    if n < 3:
        raise Exception("Error in format")

    path_to_model = sys.argv[1]
    inverted_idx = None
    with open(path_to_model, 'rb') as pickle_file:
        inverted_idx = pickle.load(pickle_file)
    
    path_to_query = sys.argv[2]
    queries = pd.read_csv(path_to_query)
    outputs = []
    for index, row in queries.iterrows():
        query = row['query'].split()
        query_id = row['topic-id']

        cord_ids = retrieveDocuments(inverted_idx, query)
        output = f"{query_id} : {' '.join(cord_ids)}"
        outputs.append(output)
    
    outputs = '\n'.join(outputs)
    with open('Assignment1_10_results.txt', "w") as f:
        f.write(outputs)
    

    
    