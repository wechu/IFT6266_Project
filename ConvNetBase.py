import theano
from theano import tensor as T
import numpy as np
import numpy.random as rng
import lasagne
from lasagne import layers as L
import pickle, gzip
import matplotlib.pyplot as plt
import time

import utils

#########
# This will be a baseline model, using only the image (no captions) to do the inpainting
#
#########
#TODO save weights
#TODO plot train/valid error graphs
#TODO add patience mechanism
#TODO check cs231n for advice
#TODO check gradient updates?



def build_cnn(input_var=None, batch_size=32):
    network = L.InputLayer(shape=(batch_size, 3, 64, 64), input_var=input_var)

    network = L.batch_norm(L.Conv2DLayer(network, num_filters=32, filter_size=(3, 3),
                                         nonlinearity=lasagne.nonlinearities.LeakyRectify(0.01),
                                         pad="same", stride=1))
    network = L.batch_norm(L.Conv2DLayer(network, num_filters=32, filter_size=(3, 3),
                                         nonlinearity=lasagne.nonlinearities.LeakyRectify(0.01),
                                         pad="same", stride=1))

    network = L.MaxPool2DLayer(network, pool_size=(2,2))

    network = L.batch_norm(L.Conv2DLayer(network, num_filters=64, filter_size=(3, 3),
                            nonlinearity=lasagne.nonlinearities.LeakyRectify(0.01),
                            pad="same", stride=1))

    network = L.batch_norm(L.Conv2DLayer(network, num_filters=64, filter_size=(3, 3),
                            nonlinearity=lasagne.nonlinearities.LeakyRectify(0.01),
                            pad="same"))

    network = L.MaxPool2DLayer(network, pool_size=(2,2))

    network = L.batch_norm(L.Conv2DLayer(network, num_filters=64, filter_size=(3, 3),
                            nonlinearity=lasagne.nonlinearities.LeakyRectify(0.01),
                            pad="same", stride=1))

    network = L.batch_norm(L.Conv2DLayer(network, num_filters=64, filter_size=(3, 3),
                            nonlinearity=lasagne.nonlinearities.LeakyRectify(0.01),
                            pad="same"))

    network = L.DenseLayer(L.dropout(network, p=0.5), num_units=3072, #L.dropout(network, p=0.5)
                           nonlinearity=lasagne.nonlinearities.LeakyRectify(0.01))

    network = L.ReshapeLayer(network, shape=(batch_size, 3, 32, 32))

    network = L.batch_norm(L.Conv2DLayer(network, num_filters=64, filter_size=(3, 3),
                            nonlinearity=lasagne.nonlinearities.LeakyRectify(0.01),
                            pad="same", stride=1))

    network = L.batch_norm(L.Conv2DLayer(network, num_filters=64, filter_size=(3, 3),
                            nonlinearity=lasagne.nonlinearities.LeakyRectify(0.01),
                            pad="same", stride=1))

    network = L.LocallyConnected2DLayer(network, num_filters=3, filter_size=(1, 1),
                            nonlinearity=lasagne.nonlinearities.LeakyRectify(0.01),
                            pad="same")

    return network



def build_cnn2(input_var=None, batch_size=32):
    # Has no fully connected layer, uses a locally connected layer at the end
    # Super slow for some reason... might be that locally connected layers are not efficient on cpu
    network = L.InputLayer(shape=(batch_size, 3, 64, 64), input_var=input_var)

    network = L.batch_norm(L.Conv2DLayer(network, num_filters=32, filter_size=(3, 3),
                                         nonlinearity=lasagne.nonlinearities.LeakyRectify(0.01),
                                         pad="same", stride=1))
    # network = L.batch_norm(L.Conv2DLayer(network, num_filters=32, filter_size=(3, 3),
    #                                      nonlinearity=lasagne.nonlinearities.LeakyRectify(0.01),
    #                                      pad="same", stride=1))

    network = L.batch_norm(L.DilatedConv2DLayer(network, num_filters=32, filter_size=(3, 3),
                                                nonlinearity=lasagne.nonlinearities.LeakyRectify(0.01),
                                                dilation=(2,2)))

    # network = L.batch_norm(L.DilatedConv2DLayer(network, num_filters=32, filter_size=(3, 3),
    #                                             nonlinearity=lasagne.nonlinearities.LeakyRectify(0.01),
    #                                             dilation=(4,4)))

    # network = L.batch_norm(L.DilatedConv2DLayer(network, num_filters=32, filter_size=(3, 3),
    #                                             nonlinearity=lasagne.nonlinearities.LeakyRectify(0.01),
    #                                             dilation=(8,8)))

    # network = L.MaxPool2DLayer(network, pool_size=(2,2))
    # # (batch, 32, 32, 32)
    #
    # network = L.batch_norm(L.Conv2DLayer(network, num_filters=32, filter_size=(3, 3),
    #                         nonlinearity=lasagne.nonlinearities.LeakyRectify(0.01),
    #                         pad="same", stride=1))
    #
    # network = L.batch_norm(L.Conv2DLayer(network, num_filters=32, filter_size=(3, 3),
    #                         nonlinearity=lasagne.nonlinearities.LeakyRectify(0.01),
    #                         pad="same"))


    # network = L.LocallyConnected2DLayer(network, num_filters=8, filter_size=(7,7),
    #                                     nonlinearity=lasagne.nonlinearities.LeakyRectify(0.01),
    #                                     pad='same')


    network = L.LocallyConnected2DLayer(network, num_filters=8, filter_size=(5,5),
                                        nonlinearity=lasagne.nonlinearities.LeakyRectify(0.01),
                                        pad='same')

    network = L.batch_norm(L.Conv2DLayer(network, num_filters=8, filter_size=(5, 5),
                                         nonlinearity=lasagne.nonlinearities.LeakyRectify(0.01),
                                         pad='valid', stride=1))

    network = L.LocallyConnected2DLayer(network, num_filters=3, filter_size=(5,5),
                                        nonlinearity=lasagne.nonlinearities.LeakyRectify(0.01),
                                        pad='same')

    return network

