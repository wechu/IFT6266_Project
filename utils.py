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


if __name__ == "__main__":
    # x, y, cap = load_dataset()
    plt.rcParams['toolbar'] = 'None'

    # print(cap[0])
    #
    # print(len(x), len(y), len(cap))
    #
    # plt.figure(figsize=(8,8))
    # plt.imshow(x[0])
    # plt.figure(figsize=(4,4))
    # plt.imshow(y[0])
    # plt.show()


