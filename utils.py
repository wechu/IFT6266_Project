import theano
import theano.tensor as T
import numpy as np
import numpy.random as rng
import matplotlib.pyplot as plt

import gzip
import pickle
import lasagne

def load_dataset(file):

    with gzip.open(file, 'rb') as pickle_file:
        input_img, target_img, caption = pickle.load(pickle_file)   
    
    input_img = np.array(input_img, dtype='float32') # rescale data to [0,1]
    input_img /= 255
    target_img = np.array(target_img, dtype='float32')
    target_img /= 255
    # Rearrange the dims in the data
    input_img = input_img.transpose((0, 3, 1, 2))  # put the channel dim before the horiz/verti dims
    target_img = target_img.transpose((0, 3, 1, 2))
    
    return input_img, target_img, caption


def save_model_weights(model, filename):
    weights = lasagne.layers.get_all_param_values(model)
    with open(filename, 'wb') as f:
        pickle.dump(weights, f)
    return

def load_model_weights(model, filename):
    with open(filename, 'rb') as f:
        weights = pickle.load(f)
    lasagne.layers.set_all_param_values(model, weights)
    return

def save_updates(model_updates, filename):
    all_values = [item.get_value() for item in model_updates]
    with open(filename, 'wb') as f:
        pickle.dump(all_values, f)
    return

def load_updates(model_updates, filename):
    # compile first though
    with open(filename, 'rb') as f:
        ups = pickle.load(f)
    for u, v in zip(model_updates, ups):
        u.set_value(v)
    return