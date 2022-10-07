import pickle
import json
import glob
import string
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer


def build_raw_doc(data_dir, id_maps):
  """
    Returns a dictionary of raw documents 
    indexed by the cord id
  """
  # Get the cord ids
  id_maps = pd.read_csv(coord_file)
  id_maps = id_maps.reset_index('paper_id')
  
  my_raw_documents = dict()
  for filename in glob.iglob(f"{data_dir}*.json", recursive = True):
    with open(filename, 'r') as f:
        data = json.load(f)
      
    # Get the paper id
    paper_id = data['paper_id']
    # Get the associated cord id
    cord_id = id_maps.loc[paper_id].cord_id
    
    # Create if doesnt exist
    if cord_id not in my_raw_documents:
      my_raw_documents[cord_id] = ""
      
    # Get all the text from the files
    for line in data['abstract']:
      my_raw_documents[cord_id]+=(" "+line['text'])
      
  return my_raw_documents


def preprocess_and_gen_tokens(text):
  """
    Takes a document and returns a set of
    tokens in the document
  """
  wordnet_lemmatizer = WordNetLemmatizer()
  # Remove punctuations
  text = text.translate(str.maketrans('', '', string.punctuation))
  # Tokenize
  tokens = word_tokenize(text)
  # Lemmatize
  tokens = [wordnet_lemmatizer.lemmatize(word) for word in tokens]
  # Return unique tokens
  tokens = list(set(tokens))
  return tokens


def build_index(my_raw_documents):
  """
    Returns an inverted index
  """ 
  inverted_idx = dict()
  
  for cord_id in sorted(my_raw_documents):
    # Get the tokens
    tokens = preprocess_and_gen_tokens(my_raw_documents[cord_id])
    # Add to the inverted index
    for token in tokens:
      if token not in inverted_idx:
        inverted_idx[token] = []
      inverted_idx[token].append(cord_id)
      
  return inverted_idx


def save_index(inverted_idx, index_file):
  """
    Saves the inverted index to a file
  """
  with open(index_file, 'wb') as f:
    pickle.dump(inverted_idx, f)


if __name__ == "__main__":
  data_dir = 'Data/CORD-19/'
  coord_file = 'Data/id_mapping.csv'
  index_file = 'model_queries_10.bin'
  
  my_raw_documents = build_raw_doc(data_dir, coord_file)
  inverted_idx = build_index(my_raw_documents)  
  save_index(inverted_idx, index_file)