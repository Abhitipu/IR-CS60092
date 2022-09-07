import pickle

#  yeh chal gaya, me wb karke uda diya lekin :(, purana wala pickle hai
# shit... 'rb' karna tha .. also context manager se close ho jaata hai
# Vinit kidhar gaya be shit true
# context manager daal raha
 
inverted_idx = pickle.load(open('model_queries_10.bin', 'rb'))

# inverted_idx mein sorted hoga? and kya store karege filename, ya cord-id
# yes sorted -- only id par file access karna padega
# 5. Build Inverted Index (Dictionary with tokens as keys, and document name as postings
# Chill hai fir, but sorted ? Strings will be sorted? pata nahi? list.sort call kardenge? thik

# with open('model_queries_10.bin', 'rb') as f:
#     inverted_index = pickle.load(f)
for index in inverted_idx.keys():
    print(inverted_idx[index])
    break