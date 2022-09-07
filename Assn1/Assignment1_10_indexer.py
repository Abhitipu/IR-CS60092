import nltk
import pickle
import os
import json
import glob
import string
from nltk.tokenize import word_tokenize
from nltk.stem import 	WordNetLemmatizer


if __name__ == "__main__":
  directory = 'Data/CORD-19/'

  inverted_idx = {}
  wordnet_lemmatizer = WordNetLemmatizer()
  cnt = 1
  
  # for filename in os.listdir(directory):
  for filename in glob.iglob(f"{directory}*.json", recursive = True):
      filepath = os.path.join(directory, filename)
      # f = open(filepath)
      # Ab chala
      # Also object oriented nahi banana?
        #   haa class karna hai
        # idhar tab me kuch bakchodi ho rahi hai
      # Class ke bina to maza nahi aayega : P
      # Haan woh to samajh aa raha
      # So index to bana liya? Ab commit and push peace
      # git add, git commit, git out
      # Aur Vinit : )  hogaya ab ? 
      f = open(filename)
      data = json.load(f)
      paper_id = data['paper_id']
      text = ""
      for line in data['abstract']:
        text+=(" "+line['text'])
      
      tokens = word_tokenize(text)
      tokens = list(filter(lambda token: token not in string.punctuation, tokens))
      tokens = [wordnet_lemmatizer.lemmatize(word) for word in tokens]
      tokens = list(set(tokens))

      for word in tokens:
        if word in inverted_idx:
          inverted_idx[word].append(paper_id)
        else:
          inverted_idx[word] = [paper_id]
      if(cnt %1000 == 0):
        print(f"Processed files {cnt} last paper_id : {paper_id}")
      cnt+=1  


  print(len(inverted_idx))
  # machaya machaya bhai waise... agar hum files ko hi rename kar dein? haa yeh sahi lag raha kal karte hai
  # Idea: id -> filename ek map rakh lete.. peace
  # one sec
  # before commit .bin ko bhi daal dena gitignore mein picke kar de raha hu upload 31 mb hai
  # accha ye chal raha hai... mujhe laga wait kyu kar rahe lol hogaya!... karde pushhhhhh mai jaa raha
  # Kaafi zyada hai na ... i mean.. koi faida hoga?no idea  # ignore hi kar dete, thik abhi commit me nahi dalta bin
  for key in inverted_idx.keys():
    inverted_idx[key].sort()
  pickle.dump(inverted_idx, open('model_queries_10.bin', 'wb'))

