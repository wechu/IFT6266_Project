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




class CNN():
    def __init__(self, name=""):
        self.name = name
        self.model = None
        pass

    def compile_theano(self, model, learn_rate, batch_size):
        input_var = T.tensor4('inputs', dtype='float32')
        target_var = T.tensor4('targets', dtype='float32')

        # initialize model
        if model == "v1":
            self.model = self.build_cnn(input_var, batch_size)
        elif model == "v2":
            self.model = self.build_cnn2(input_var, batch_size)
        elif model == "v3":
            self.model = self.build_cnn3(input_var, batch_size)
        elif model == "v4":
            self.model = self.build_cnn4(input_var, batch_size)
        else:
            raise Exception("No model of that name.")

        # get model
        network = self.model

        # get prediction and loss
        pred = L.get_output(network)
        loss = lasagne.objectives.squared_error(pred, target_var).mean()

        # get test predictions and loss
        test_pred = L.get_output(network, deterministic=True)
        test_loss = lasagne.objectives.squared_error(test_pred, target_var)
        test_loss = test_loss.mean()

        # get updates
        lr = theano.shared(np.array(learn_rate, dtype='float32'))

        params = L.get_all_params(network, trainable=True)
        updates = lasagne.updates.adam(loss, params, learning_rate=lr)

        print("Compiling methods...")
        # compile methods
        self.train_method = theano.function([input_var, target_var], [loss, pred], updates=updates)
        self.valid_method = theano.function([input_var, target_var], [test_loss, test_pred])

        return

    def build_cnn(self, input_var=None, batch_size=64):
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

    def build_cnn2(self, input_var=None, batch_size=64):
        # autoencoder-type architecture
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


        network = L.batch_norm(L.Conv2DLayer(network, num_filters=96, filter_size=(3, 3),
                                             nonlinearity=lasagne.nonlinearities.LeakyRectify(0.01),
                                             pad="same", stride=1))

        network = L.batch_norm(L.Conv2DLayer(network, num_filters=96, filter_size=(3, 3),
                                             nonlinearity=lasagne.nonlinearities.LeakyRectify(0.01),
                                             pad="same"))

        network = L.MaxPool2DLayer(network, pool_size=(2,2))

        network = L.batch_norm(L.Conv2DLayer(network, num_filters=128, filter_size=(3, 3),
                                             nonlinearity=lasagne.nonlinearities.LeakyRectify(0.01),
                                             pad="same", stride=1))

        network = L.batch_norm(L.Conv2DLayer(network, num_filters=128, filter_size=(3, 3),
                                             nonlinearity=lasagne.nonlinearities.LeakyRectify(0.01),
                                             pad="same"))
        network = L.MaxPool2DLayer(network, pool_size=(2, 2))

        network = L.batch_norm(L.Conv2DLayer(network, num_filters=192, filter_size=(3, 3),
                                             nonlinearity=lasagne.nonlinearities.LeakyRectify(0.01),
                                             pad="same"))

        network = L.batch_norm(L.Conv2DLayer(network, num_filters=192, filter_size=(3, 3),
                                             nonlinearity=lasagne.nonlinearities.LeakyRectify(0.01),
                                             pad="same"))

        network = L.DenseLayer(L.dropout(network, p=0.5), num_units=512,  # L.dropout(network, p=0.5)
                               nonlinearity=lasagne.nonlinearities.LeakyRectify(0.01))

        network = L.ReshapeLayer(network, shape=(batch_size, 8, 8, 8))

        network = L.batch_norm(L.Conv2DLayer(network, num_filters=192, filter_size=(3, 3),
                                             nonlinearity=lasagne.nonlinearities.LeakyRectify(0.01),
                                             pad="same", stride=1))

        network = L.batch_norm(L.Conv2DLayer(network, num_filters=192, filter_size=(3, 3),
                                             nonlinearity=lasagne.nonlinearities.LeakyRectify(0.01),
                                             pad="same", stride=1))

        network = L.Upscale2DLayer(network, 2, mode='repeat')

        network = L.batch_norm(L.Conv2DLayer(network, num_filters=128, filter_size=(3, 3),
                                             nonlinearity=lasagne.nonlinearities.LeakyRectify(0.01),
                                             pad="same", stride=1))

        network = L.batch_norm(L.Conv2DLayer(network, num_filters=128, filter_size=(3, 3),
                                             nonlinearity=lasagne.nonlinearities.LeakyRectify(0.01),
                                             pad="same", stride=1))

        network = L.Upscale2DLayer(network, 2, mode='repeat')

        network = L.batch_norm(L.Conv2DLayer(network, num_filters=128, filter_size=(3, 3),
                                             nonlinearity=lasagne.nonlinearities.LeakyRectify(0.01),
                                             pad="same", stride=1))

        network = L.batch_norm(L.Conv2DLayer(network, num_filters=128, filter_size=(3, 3),
                                             nonlinearity=lasagne.nonlinearities.LeakyRectify(0.01),
                                             pad="same", stride=1))

        network = L.LocallyConnected2DLayer(network, num_filters=3, filter_size=(1, 1),
                                            nonlinearity=lasagne.nonlinearities.LeakyRectify(0.01),
                                            pad="same")
        return network

    def build_cnn3(self, input_var=None, batch_size=64):
        # autoencoder-type architecture
        # oops downsample too much (forgot to account for kernel size)
        network = L.InputLayer(shape=(batch_size, 3, 64, 64), input_var=input_var)

        network = L.batch_norm(L.Conv2DLayer(network, num_filters=64, filter_size=(3, 3),
                                             nonlinearity=lasagne.nonlinearities.LeakyRectify(0.01),
                                             pad="same", stride=1))
        network = L.batch_norm(L.Conv2DLayer(network, num_filters=64, filter_size=(3, 3),
                                             nonlinearity=lasagne.nonlinearities.LeakyRectify(0.01),
                                             pad="same", stride=1))

        network = L.batch_norm(L.Conv2DLayer(network, num_filters=64, filter_size=(3, 3),
                                             nonlinearity=lasagne.nonlinearities.LeakyRectify(0.01),
                                             pad=0, stride=2))
        network = L.batch_norm(L.Conv2DLayer(network, num_filters=64, filter_size=(3, 3),
                                             nonlinearity=lasagne.nonlinearities.LeakyRectify(0.01),
                                             pad="same", stride=1))
        network = L.batch_norm(L.Conv2DLayer(network, num_filters=64, filter_size=(3, 3),
                                             nonlinearity=lasagne.nonlinearities.LeakyRectify(0.01),
                                             pad="same"))

        network = L.batch_norm(L.Conv2DLayer(network, num_filters=96, filter_size=(3, 3),
                                             nonlinearity=lasagne.nonlinearities.LeakyRectify(0.01),
                                             pad=0, stride=2))
        network = L.batch_norm(L.Conv2DLayer(network, num_filters=96, filter_size=(3, 3),
                                             nonlinearity=lasagne.nonlinearities.LeakyRectify(0.01),
                                             pad="same", stride=1))
        network = L.batch_norm(L.Conv2DLayer(network, num_filters=96, filter_size=(3, 3),
                                             nonlinearity=lasagne.nonlinearities.LeakyRectify(0.01),
                                             pad="same"))


        network = L.batch_norm(L.Conv2DLayer(network, num_filters=128, filter_size=(3, 3),
                                             nonlinearity=lasagne.nonlinearities.LeakyRectify(0.01),
                                             pad=0, stride=2))
        network = L.batch_norm(L.Conv2DLayer(network, num_filters=128, filter_size=(3, 3),
                                             nonlinearity=lasagne.nonlinearities.LeakyRectify(0.01),
                                             pad="same", stride=1))
        network = L.batch_norm(L.Conv2DLayer(network, num_filters=128, filter_size=(3, 3),
                                             nonlinearity=lasagne.nonlinearities.LeakyRectify(0.01),
                                             pad="same"))

        network = L.batch_norm(L.Conv2DLayer(network, num_filters=256, filter_size=(3, 3),
                                             nonlinearity=lasagne.nonlinearities.LeakyRectify(0.01),
                                             pad=0, stride=2))
        network = L.batch_norm(L.Conv2DLayer(network, num_filters=256, filter_size=(3, 3),
                                             nonlinearity=lasagne.nonlinearities.LeakyRectify(0.01),
                                             pad="same"))
        network = L.batch_norm(L.Conv2DLayer(network, num_filters=256, filter_size=(3, 3),
                                             nonlinearity=lasagne.nonlinearities.LeakyRectify(0.01),
                                             pad="same"))
        # 3x3 here
        network = L.DenseLayer(L.dropout(network, p=0.5), num_units=2048,  # L.dropout(network, p=0.5)
                               nonlinearity=lasagne.nonlinearities.LeakyRectify(0.01))

        network = L.ReshapeLayer(network, shape=(batch_size, 128, 4, 4))

        network = L.batch_norm(L.LocallyConnected2DLayer(network, num_filters=128, filter_size=(3, 3),
                                                         nonlinearity=lasagne.nonlinearities.LeakyRectify(0.01),
                                                         pad="same", channelwise=True))
        network = L.batch_norm(L.LocallyConnected2DLayer(network, num_filters=128, filter_size=(3, 3),
                                                         nonlinearity=lasagne.nonlinearities.LeakyRectify(0.01),
                                                         pad="same", channelwise=True))

        network = L.batch_norm(L.Conv2DLayer(network, num_filters=256, filter_size=(3, 3),
                                             nonlinearity=lasagne.nonlinearities.LeakyRectify(0.01),
                                             pad="same", stride=1))
        network = L.batch_norm(L.Conv2DLayer(network, num_filters=256, filter_size=(3, 3),
                                             nonlinearity=lasagne.nonlinearities.LeakyRectify(0.01),
                                             pad="same", stride=1))

        network = L.Upscale2DLayer(network, 2, mode='repeat')
        network = L.batch_norm(L.Conv2DLayer(network, num_filters=128, filter_size=(3, 3),
                                             nonlinearity=lasagne.nonlinearities.LeakyRectify(0.01),
                                             pad="same", stride=1))
        network = L.batch_norm(L.Conv2DLayer(network, num_filters=128, filter_size=(3, 3),
                                             nonlinearity=lasagne.nonlinearities.LeakyRectify(0.01),
                                             pad="same", stride=1))

        network = L.Upscale2DLayer(network, 2, mode='repeat')
        network = L.batch_norm(L.Conv2DLayer(network, num_filters=64, filter_size=(3, 3),
                                             nonlinearity=lasagne.nonlinearities.LeakyRectify(0.01),
                                             pad="same", stride=1))
        network = L.batch_norm(L.Conv2DLayer(network, num_filters=64, filter_size=(3, 3),
                                             nonlinearity=lasagne.nonlinearities.LeakyRectify(0.01),
                                             pad="same", stride=1))

        network = L.Upscale2DLayer(network, 2, mode='repeat')
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


    def build_cnn4(self, input_var=None, batch_size=64):
        # autoencoder-type architecture
        # fixed downsampling (pad=1), made FC layer larger
        network = L.InputLayer(shape=(batch_size, 3, 64, 64), input_var=input_var)

        network = L.batch_norm(L.Conv2DLayer(network, num_filters=64, filter_size=(3, 3),
                                             nonlinearity=lasagne.nonlinearities.LeakyRectify(0.01),
                                             pad="same", stride=1))
        network = L.batch_norm(L.Conv2DLayer(network, num_filters=64, filter_size=(3, 3),
                                             nonlinearity=lasagne.nonlinearities.LeakyRectify(0.01),
                                             pad="same", stride=1))

        network = L.batch_norm(L.Conv2DLayer(network, num_filters=64, filter_size=(3, 3),
                                             nonlinearity=lasagne.nonlinearities.LeakyRectify(0.01),
                                             pad=1, stride=2))
        network = L.batch_norm(L.Conv2DLayer(network, num_filters=64, filter_size=(3, 3),
                                             nonlinearity=lasagne.nonlinearities.LeakyRectify(0.01),
                                             pad="same", stride=1))
        network = L.batch_norm(L.Conv2DLayer(network, num_filters=64, filter_size=(3, 3),
                                             nonlinearity=lasagne.nonlinearities.LeakyRectify(0.01),
                                             pad="same"))

        network = L.batch_norm(L.Conv2DLayer(network, num_filters=128, filter_size=(3, 3),
                                             nonlinearity=lasagne.nonlinearities.LeakyRectify(0.01),
                                             pad=1, stride=2))
        network = L.batch_norm(L.Conv2DLayer(network, num_filters=128, filter_size=(3, 3),
                                             nonlinearity=lasagne.nonlinearities.LeakyRectify(0.01),
                                             pad="same", stride=1))
        network = L.batch_norm(L.Conv2DLayer(network, num_filters=128, filter_size=(3, 3),
                                             nonlinearity=lasagne.nonlinearities.LeakyRectify(0.01),
                                             pad="same"))


        network = L.batch_norm(L.Conv2DLayer(network, num_filters=128, filter_size=(3, 3),
                                             nonlinearity=lasagne.nonlinearities.LeakyRectify(0.01),
                                             pad=1, stride=2))
        network = L.batch_norm(L.Conv2DLayer(network, num_filters=128, filter_size=(3, 3),
                                             nonlinearity=lasagne.nonlinearities.LeakyRectify(0.01),
                                             pad="same", stride=1))
        network = L.batch_norm(L.Conv2DLayer(network, num_filters=128, filter_size=(3, 3),
                                             nonlinearity=lasagne.nonlinearities.LeakyRectify(0.01),
                                             pad="same"))

        network = L.batch_norm(L.Conv2DLayer(network, num_filters=256, filter_size=(3, 3),
                                             nonlinearity=lasagne.nonlinearities.LeakyRectify(0.01),
                                             pad=1, stride=2))
        network = L.batch_norm(L.Conv2DLayer(network, num_filters=256, filter_size=(3, 3),
                                             nonlinearity=lasagne.nonlinearities.LeakyRectify(0.01),
                                             pad="same"))
        network = L.batch_norm(L.Conv2DLayer(network, num_filters=256, filter_size=(3, 3),
                                             nonlinearity=lasagne.nonlinearities.LeakyRectify(0.01),
                                             pad="same"))
        # 4x4 image size here
        network = L.batch_norm(L.DenseLayer(L.dropout(network, p=0.5), num_units=3072,
                                            nonlinearity=lasagne.nonlinearities.LeakyRectify(0.01)))

        network = L.ReshapeLayer(network, shape=(batch_size, 192, 4, 4))

        network = L.batch_norm(L.LocallyConnected2DLayer(network, num_filters=192, filter_size=(3, 3),
                                                         nonlinearity=lasagne.nonlinearities.LeakyRectify(0.01),
                                                         pad="same", channelwise=True))

        network = L.batch_norm(L.Conv2DLayer(network, num_filters=256, filter_size=(3, 3),
                                             nonlinearity=lasagne.nonlinearities.LeakyRectify(0.01),
                                             pad="same", stride=1))
        network = L.batch_norm(L.Conv2DLayer(network, num_filters=256, filter_size=(3, 3),
                                             nonlinearity=lasagne.nonlinearities.LeakyRectify(0.01),
                                             pad="same", stride=1))

        network = L.Upscale2DLayer(network, 2, mode='repeat')
        network = L.batch_norm(L.Conv2DLayer(network, num_filters=128, filter_size=(3, 3),
                                             nonlinearity=lasagne.nonlinearities.LeakyRectify(0.01),
                                             pad="same", stride=1))
        network = L.batch_norm(L.Conv2DLayer(network, num_filters=128, filter_size=(3, 3),
                                             nonlinearity=lasagne.nonlinearities.LeakyRectify(0.01),
                                             pad="same", stride=1))

        network = L.Upscale2DLayer(network, 2, mode='repeat')
        network = L.batch_norm(L.Conv2DLayer(network, num_filters=96, filter_size=(3, 3),
                                             nonlinearity=lasagne.nonlinearities.LeakyRectify(0.01),
                                             pad="same", stride=1))
        network = L.batch_norm(L.Conv2DLayer(network, num_filters=96, filter_size=(3, 3),
                                             nonlinearity=lasagne.nonlinearities.LeakyRectify(0.01),
                                             pad="same", stride=1))

        network = L.Upscale2DLayer(network, 2, mode='repeat')
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
