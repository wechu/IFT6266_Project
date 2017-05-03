### Note: USe python 2.7 for this stuff
# Used to save captions as thought vectors (ie. sentence embeddings)
import gzip
import cPickle as pickle
import imp
import sys
import time
import gc
import numpy as np
import itertools
st = imp.load_source("skipthoughts", "C:/Users/Wes/Documents/DL Project/skip-thoughts-master/skipthoughts.py")
#
# # Get encoder model
# model = st.load_model()
# encoder = st.Encoder(model)
#
# # Get data
# file = "Data/val2014.pkl.gz"
# with gzip.open(file, 'rb') as pickle_file:
#     border, center, caption = pickle.load(pickle_file)
# del border
# del center
# gc.collect()
# ##save number of examples
# num_x = len(caption)
# print num_x
#
# # take only 5 captions from each example
# caption = [example[:5] for example in caption]
#
# # for example in caption:
# #     if len(example) > 5:
# #         print "-----" + str(len(example))
# #     sys.stdout.write(str(len(example)) + " ")
#
# # flatten list
# caption = [sentence for example in caption for sentence in example]
#
# print "Start processing"
# # get embeddings
# m = 500
# total_iters = int(len(caption) / m) +1
# iter_per_macro = 30
# print "Total iters", total_iters
# print "Total length", len(caption)
# print "Number macro batches", int(total_iters / iter_per_macro)+1
#
# it = 0
# start = time.clock()
#
# # Process embedding in ba
# # tches since it can't all fit in memory
# for macro_i in range(13, 14):
#     print ("-x-x-x-x-" + str(macro_i))
#     embedding = []  # reset embeddings
#
#     for i in range(macro_i*iter_per_macro, (macro_i+1)*iter_per_macro):
#         print ("----- " + str(i))
#         embedding.append(encoder.encode(caption[m*i:m*(i+1)]))
#
#         if i >= total_iters:
#             print "Stopping here"
#             break
#     # Put every embedding into one big list (to avoid lists of lists)
#     embedding = np.concatenate(embedding)
#     print "saving..."
#     with open("Data/val2014_embed%s.pkl" % str(macro_i), 'wb') as pickle_file:
#         pickle.dump(embedding, pickle_file, protocol=2)
# print time.clock() - start
#
# print "done"
# ############################## Phase 2 : Merge all the files
# all_embeddings = []
#
# for j in range(14):
#     file_name = "Data/val2014_embed%s.pkl" % str(j)
#     print file_name
#     with open(file_name, 'rb') as pickle_file:
#         loaded = pickle.load(pickle_file)
#         all_embeddings.append(loaded)
#
# # group all batches of embeddings together
# all_embeddings = np.concatenate(all_embeddings)
# print all_embeddings.shape
# # regroup by lists of 5 embeddings per image
# all_embeddings = np.split(all_embeddings, 40438) # number of examples originally
# print "saving..."
# # Save to one file
# with open("Data/val2014_embed_all.pkl", 'wb') as pickle_file:
#     pickle.dump(all_embeddings, pickle_file, protocol=2)

######################## Test if it works

# file_name = "Data/train2014_embed_all.pkl.gz"
# with gzip.open(file_name, 'rb') as pickle_file:
#     loaded = pickle.load(pickle_file)
#
# with open("Data/train2014_embed_all.npy", 'wb') as pickle_file:
#     pickle.dump(np.array(loaded, dtype='float32'), pickle_file, protocol=2)
