import pickle

if __name__ == "__main__":
    filename = "model_queries_10.bin"
    
    with open(filename, r) as f:
        inv_idx = pickle.load(f)
    
    # 1. Df(t) -- size of list
    # 2. TF(t, d) 
    
    pass