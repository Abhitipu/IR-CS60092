import pandas as pd
import string
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import sys

def preprocess_and_gen_tokens(text):
  """
    Takes a document and returns a set of
    tokens in the document AS A STRING
  """
  wordnet_lemmatizer = WordNetLemmatizer()
  # Case Normalization
  text = text.lower()
  # Remove punctuations
  text = text.translate(str.maketrans('', '', string.punctuation))
  # Tokenize
  tokens = word_tokenize(text)
  # Lemmatize
  tokens = [wordnet_lemmatizer.lemmatize(word) for word in tokens]
  # Return unique tokens
  tokens = list(set(tokens))
  return " ".join(tokens)

if __name__ == "__main__":
  n = len(sys.argv)
  
  if n < 2:
    raise Exception("Error in format")
  
  query_file = sys.argv[1]
  result_file = "queries_10.txt"
  
  queries = pd.read_csv(query_file)
  queries.drop(['question', 'narrative'], axis=1, inplace=True)
  queries['query'] = queries['query'].map(lambda x: preprocess_and_gen_tokens(x))
  queries.to_csv("queries_10.txt", index=False)