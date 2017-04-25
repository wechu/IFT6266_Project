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


class GAN():
    def __init__(self, name="", noise_dim=10):
        self.name = name
        self.noise_dim = noise_dim
        self.gen = None  # generator
        self.dis = None  # discriminator
        pass

    def compile_theano(self, model, learn_rate, batch_size):
        input_var = T.tensor4('input discriminator', dtype='float32')
        noise_var = T.matrix('noise', dtype='float32')
        target_var = T.tensor4('targets', dtype='float32')

        # initialize model
        if model == "GAN1":
            self.gen = self.build_gen(noise_var, batch_size)
            self.dis = self.build_dis(input_var, batch_size)
        else:
            raise Exception("No model of that name.")

        # get predictions from discriminator

        real_prob = L.get_output(self.dis)
        fake_prob = L.get_output(self.dis, inputs=L.get_output(self.gen))

        loss_dis = -T.mean(T.log(real_prob) + T.log(1-fake_prob))
        loss_gen = -T.mean(T.log(fake_prob))

        # get updates
        lr = theano.shared(np.array(learn_rate, dtype='float32'))

        params_gen = L.get_all_params(self.gen, trainable=True)
        params_dis = L.get_all_params(self.dis, trainable=True)

        updates_gen = lasagne.updates.adam(loss_gen, params_gen, learning_rate=lr, beta1=0.5)
        updates_dis = lasagne.updates.momentum(loss_dis, params_dis, learning_rate=lr)

        print("Compiling methods...")
        # compile methods to train
        # trains gen with dis fixed
        self.train_gen_method = theano.function([noise_var], loss_gen, updates=updates_gen)
        # trains dis with both real and fake data (gen fixed)
        self.train_dis_method = theano.function([noise_var, input_var], loss_dis, updates=updates_dis)
        # method to generate just images
        self.gen_method = theano.function([noise_var], L.get_output(self.gen, deterministic=True))

        return

    def train(self, targets, inputs=None, caps=None):
        batch_size=64
        log_file = 'ganlog.txt'
        x_train_num = targets.shape[0]
        # training loop
        num_epochs = 50
        iter = 1
        index_list = np.arange(x_train_num, dtype='int32')  # used for shuffling
        gen_file = self.name + "_gen.txt"
        dis_file = self.name + "_dis.txt"

        print("Start training...")
        time_start = time.time()
        for epoch in range(num_epochs):
            with open(log_file, 'a') as f:
                f.write("----- Epoch" + str(epoch) + " " + str(time.time() - time_start) + " -----\n")

            # Shuffle examples every epoch
            np.random.shuffle(index_list)
            # generate new noise
            noise = np.random.normal(0, 1, size=(targets.shape[0], self.noise_dim)).astype('float32')

            for i in range(int(x_train_num / batch_size)):
                shuffle_ind = index_list[i * batch_size:(i + 1) * batch_size]

                # alternate between discriminator and generator
                loss_gen = self.train_gen_method(noise[shuffle_ind])
                loss_dis = self.train_dis_method(noise[shuffle_ind], targets[shuffle_ind])

                with open(log_file, 'a') as f:
                    f.write(str(epoch) + " " + str(i) + "\tDis:" + str(float(loss_dis)) + "\tGen:" + str(float(loss_gen))+ "\n")
                with open(gen_file, 'a') as f:
                    f.write(str(float(loss_gen)) + "\n")
                with open(dis_file, 'a') as f:
                    f.write(str(float(loss_dis)) + "\n")

                # Validation set and save weights
                if iter % 600 == 0 or iter == 1:
                    utils.save_model_weights(self.gen, "{}_gen_weights_e_{}_it_{}.pkl".format(self.name, epoch, iter))
                    utils.save_model_weights(self.dis, "{}_dis_weights_e_{}_it_{}.pkl".format(self.name, epoch, iter))

                iter += 1

        print("Total time", time.time() - time_start)
        # test a few values

        return

    def build_gen(self, input_gen, batch_size):
        network = L.InputLayer(shape=(batch_size, self.noise_dim), input_var=input_gen)

        network = L.DenseLayer(network, num_units=3072,
                               nonlinearity=lasagne.nonlinearities.LeakyRectify(0.01))

        network = L.ReshapeLayer(network, shape=(batch_size, 192, 4, 4))

        network = L.batch_norm(L.Conv2DLayer(network, num_filters=512, filter_size=(3, 3),
                                             nonlinearity=lasagne.nonlinearities.LeakyRectify(0.01),
                                             pad="same", stride=1))
        network = L.batch_norm(L.Conv2DLayer(network, num_filters=512, filter_size=(3, 3),
                                             nonlinearity=lasagne.nonlinearities.LeakyRectify(0.01),
                                             pad="same", stride=1))

        network = L.Upscale2DLayer(network, 2, mode='repeat')
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

        network = L.LocallyConnected2DLayer(network, num_filters=3, filter_size=(1, 1),
                                            nonlinearity=lasagne.nonlinearities.LeakyRectify(0.01),
                                            pad="same")
        # output is 32 x 32
        return network

    def build_dis(self, input_dis, batch_size):

        network = L.InputLayer(shape=(batch_size, 3, 32, 32), input_var=input_dis)

        network = L.batch_norm(L.Conv2DLayer(network, num_filters=64, filter_size=(3, 3),
                                             nonlinearity=lasagne.nonlinearities.LeakyRectify(0.01),
                                             pad="same", stride=1))
        network = L.batch_norm(L.Conv2DLayer(network, num_filters=64, filter_size=(3, 3),
                                             nonlinearity=lasagne.nonlinearities.LeakyRectify(0.01),
                                             pad="same", stride=1))

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

        network = L.batch_norm(L.Conv2DLayer(network, num_filters=512, filter_size=(3, 3),
                                             nonlinearity=lasagne.nonlinearities.LeakyRectify(0.01),
                                             pad=1, stride=2))
        network = L.batch_norm(L.Conv2DLayer(network, num_filters=512, filter_size=(3, 3),
                                             nonlinearity=lasagne.nonlinearities.LeakyRectify(0.01),
                                             pad="same"))
        network = L.batch_norm(L.Conv2DLayer(network, num_filters=512, filter_size=(3, 3),
                                             nonlinearity=lasagne.nonlinearities.LeakyRectify(0.01),
                                             pad="same"))

        network = L.DenseLayer(network, num_units=1, nonlinearity=lasagne.nonlinearities.sigmoid)

        return network



class BEGAN():
    def __init__(self, name="", noise_dim=64, gamma=0.5, k_initial=0):
        self.name = name
        self.noise_dim = noise_dim
        self.gen = None  # generator
        self.dis = None  # discriminator
        self.gamma = gamma
        self.k = k_initial
        self.updates = None
        pass

    def compile_theano(self, learn_rate, batch_size, mode):
        input_var = T.tensor4('input', dtype='float32')
        noise_var = T.matrix('noise', dtype='float32')

        # initialize model
        if mode == "v1":
            self.gen = self.build_gen(noise_var, batch_size)
            self.dis = self.build_dis(input_var, batch_size)  # discriminator is actually an autoencoder
        elif mode == "v2":  # bigger version of v1
            self.gen = self.build_gen2(noise_var, batch_size)
            self.dis = self.build_dis2(input_var, batch_size)

        # get predictions from discriminator (autoencoder)
        real_autoenc = L.get_output(self.dis)
        fake_image = L.get_output(self.gen)
        fake_autoenc = L.get_output(self.dis, inputs=fake_image)

        # use L1 reconstruction loss
        k = theano.shared(np.array(self.k, dtype='float32'))    # balancing variable
        loss_real = T.abs_(real_autoenc - input_var)
        loss_fake = T.abs_(fake_autoenc - fake_image)

        loss_dis = T.mean(loss_real - k * loss_fake)
        loss_gen = T.mean(loss_fake)

        # get updates
        lr = theano.shared(np.array(learn_rate, dtype='float32'))

        params_gen = L.get_all_params(self.gen, trainable=True)
        params_dis = L.get_all_params(self.dis, trainable=True)

        updates = lasagne.updates.adam(loss_gen, params_gen, learning_rate=lr)
        updates.update(lasagne.updates.momentum(loss_dis, params_dis, learning_rate=lr))
        updates.update({k: k + 0.001 * T.mean(self.gamma * loss_real - loss_fake)})
        self.updates = updates

        print("Compiling methods...")
        # compile methods to train (trains everything at once)
        self.train_method = theano.function([noise_var, input_var], [T.mean(loss_real), T.mean(loss_fake), k], updates=updates)

        # method to generate just images
        self.gen_method = theano.function([noise_var], L.get_output(self.gen, deterministic=True))

        return

    def compute_global_loss(self, loss_real, loss_fake):
        ''' Computes global loss to assess training '''
        return loss_real + abs(self.gamma * loss_real - loss_fake)

    def train(self, targets, batch_size=64, noise_type="gaussian", start_iter=1, log_file=None):
        save_freq = 600
        x_train_num = targets.shape[0]
        gen_file = self.name + "_gen.txt"
        dis_file = self.name + "_dis.txt"
        check_file = self.name + "_check.txt"

        # training loop
        num_epochs = 50
        iter = start_iter
        index_list = np.arange(x_train_num, dtype='int32')  # used for shuffling

        print("Start training...")
        time_start = time.time()
        for epoch in range(num_epochs):
            with open(log_file, 'a') as f:
                f.write("----- Epoch" + str(epoch) + " " + str(time.time() - time_start) + " -----\n")

            # Shuffle examples every epoch
            np.random.shuffle(index_list)
            # generate new noise
            if noise_type == "gaussian":
                noise = np.random.normal(0, 1, size=(targets.shape[0], self.noise_dim)).astype('float32')
            elif noise_type == "uniform":
                noise = np.random.uniform(-1, 1, size=(targets.shape[0], self.noise_dim)).astype('float32')

            for i in range(int(x_train_num / batch_size)):
                shuffle_ind = index_list[i * batch_size:(i + 1) * batch_size]

                # alternate between discriminator and generator
                loss_real, loss_fake, k = self.train_method(noise[shuffle_ind], targets[shuffle_ind])

                with open(log_file, 'a') as f:
                    f.write(str(epoch) + " " + str(i) + "\tk: {:.3f} Check:{:.6f} Real:{:.6f} Fake:{:.6f}\n".format(float(np.array(k)),
                        float(self.compute_global_loss(loss_real, loss_fake)), float(loss_real), float(loss_fake)))

                with open(gen_file, 'a') as f:
                    f.write(str(float(loss_fake)) + "\n")
                with open(dis_file, 'a') as f:
                    f.write(str(float(loss_real - k * loss_fake)) + "\n")
                with open(check_file, 'a') as f:
                    f.write(str(self.compute_global_loss(loss_real, loss_fake)) + "\n")

                # Validation set and save weights
                if iter % save_freq == 0 or iter == 1:
                    utils.save_model_weights(self.gen,
                                             "{}_gen_weights_e_{}_it_{}.pkl".format(self.name, epoch, iter))
                    utils.save_model_weights(self.dis,
                                             "{}_dis_weights_e_{}_it_{}.pkl".format(self.name, epoch, iter))
                    utils.save_updates(self.updates, "{}_updates_e_{}_it_{}.pkl".format(self.name, epoch, iter))

                iter += 1

        print("Total time", time.time() - time_start)
        # test a few values

        return

    def build_gen(self, input_gen, batch_size):
        network = L.InputLayer(shape=(batch_size, self.noise_dim), input_var=input_gen)

        network = L.DenseLayer(network, num_units=4096,
                               nonlinearity=lasagne.nonlinearities.identity)

        network = L.ReshapeLayer(network, shape=(batch_size, 256, 4, 4))

        network = L.batch_norm(L.Conv2DLayer(network, num_filters=64, filter_size=(3, 3),
                                             nonlinearity=lasagne.nonlinearities.elu,
                                             pad="same", stride=1))
        network = L.batch_norm(L.Conv2DLayer(network, num_filters=64, filter_size=(3, 3),
                                             nonlinearity=lasagne.nonlinearities.elu,
                                             pad="same", stride=1))

        network = L.Upscale2DLayer(network, 2, mode='repeat')
        network = L.batch_norm(L.Conv2DLayer(network, num_filters=64, filter_size=(3, 3),
                                             nonlinearity=lasagne.nonlinearities.elu,
                                             pad="same", stride=1))
        network = L.batch_norm(L.Conv2DLayer(network, num_filters=64, filter_size=(3, 3),
                                             nonlinearity=lasagne.nonlinearities.elu,
                                             pad="same", stride=1))

        network = L.Upscale2DLayer(network, 2, mode='repeat')
        network = L.batch_norm(L.Conv2DLayer(network, num_filters=64, filter_size=(3, 3),
                                             nonlinearity=lasagne.nonlinearities.elu,
                                             pad="same", stride=1))
        network = L.batch_norm(L.Conv2DLayer(network, num_filters=64, filter_size=(3, 3),
                                             nonlinearity=lasagne.nonlinearities.elu,
                                             pad="same", stride=1))

        network = L.Upscale2DLayer(network, 2, mode='repeat')
        network = L.batch_norm(L.Conv2DLayer(network, num_filters=64, filter_size=(3, 3),
                                             nonlinearity=lasagne.nonlinearities.elu,
                                             pad="same", stride=1))
        network = L.batch_norm(L.Conv2DLayer(network, num_filters=64, filter_size=(3, 3),
                                             nonlinearity=lasagne.nonlinearities.elu,
                                             pad="same", stride=1))

        network = L.Conv2DLayer(network, num_filters=3, filter_size=(3, 3),
                                nonlinearity=lasagne.nonlinearities.leaky_rectify,
                                pad="same", stride=1)

        # output is 32 x 32
        return network

    def build_dis(self, input_dis, batch_size):
        # discriminator is an autoencoder in this case
        network = L.InputLayer(shape=(batch_size, 3, 32, 32), input_var=input_dis)

        network = L.batch_norm(L.Conv2DLayer(network, num_filters=64, filter_size=(3, 3),
                                             nonlinearity=lasagne.nonlinearities.elu,
                                             pad="same", stride=1))
        network = L.batch_norm(L.Conv2DLayer(network, num_filters=64, filter_size=(3, 3),
                                             nonlinearity=lasagne.nonlinearities.elu,
                                             pad="same", stride=1))

        network = L.batch_norm(L.Conv2DLayer(network, num_filters=96, filter_size=(3, 3),
                                             nonlinearity=lasagne.nonlinearities.elu,
                                             pad=1, stride=2))
        network = L.batch_norm(L.Conv2DLayer(network, num_filters=96, filter_size=(3, 3),
                                             nonlinearity=lasagne.nonlinearities.elu,
                                             pad="same", stride=1))
        network = L.batch_norm(L.Conv2DLayer(network, num_filters=96, filter_size=(3, 3),
                                             nonlinearity=lasagne.nonlinearities.elu,
                                             pad="same"))

        network = L.batch_norm(L.Conv2DLayer(network, num_filters=128, filter_size=(3, 3),
                                             nonlinearity=lasagne.nonlinearities.elu,
                                             pad=1, stride=2))
        network = L.batch_norm(L.Conv2DLayer(network, num_filters=128, filter_size=(3, 3),
                                             nonlinearity=lasagne.nonlinearities.elu,
                                             pad="same", stride=1))
        network = L.batch_norm(L.Conv2DLayer(network, num_filters=128, filter_size=(3, 3),
                                             nonlinearity=lasagne.nonlinearities.elu,
                                             pad="same"))

        network = L.batch_norm(L.Conv2DLayer(network, num_filters=192, filter_size=(3, 3),
                                             nonlinearity=lasagne.nonlinearities.elu,
                                             pad=1, stride=2))
        network = L.batch_norm(L.Conv2DLayer(network, num_filters=192, filter_size=(3, 3),
                                             nonlinearity=lasagne.nonlinearities.elu,
                                             pad="same", stride=1))
        network = L.batch_norm(L.Conv2DLayer(network, num_filters=192, filter_size=(3, 3),
                                             nonlinearity=lasagne.nonlinearities.elu,
                                             pad="same"))

        network = L.batch_norm(L.Conv2DLayer(network, num_filters=256, filter_size=(3, 3),
                                             nonlinearity=lasagne.nonlinearities.elu,
                                             pad=1, stride=2))
        network = L.batch_norm(L.Conv2DLayer(network, num_filters=256, filter_size=(3, 3),
                                             nonlinearity=lasagne.nonlinearities.elu,
                                             pad="same"))
        network = L.batch_norm(L.Conv2DLayer(network, num_filters=256, filter_size=(3, 3),
                                             nonlinearity=lasagne.nonlinearities.elu,
                                             pad="same"))
        # 4x4 image size here
        network = L.DenseLayer(network, num_units=4096,
                               nonlinearity=lasagne.nonlinearities.identity)

        network = L.ReshapeLayer(network, shape=(batch_size, 256, 4, 4))

        network = L.batch_norm(L.Conv2DLayer(network, num_filters=64, filter_size=(3, 3),
                                             nonlinearity=lasagne.nonlinearities.elu,
                                             pad="same", stride=1))
        network = L.batch_norm(L.Conv2DLayer(network, num_filters=64, filter_size=(3, 3),
                                             nonlinearity=lasagne.nonlinearities.elu,
                                             pad="same", stride=1))

        network = L.Upscale2DLayer(network, 2, mode='repeat')
        network = L.batch_norm(L.Conv2DLayer(network, num_filters=64, filter_size=(3, 3),
                                             nonlinearity=lasagne.nonlinearities.elu,
                                             pad="same", stride=1))
        network = L.batch_norm(L.Conv2DLayer(network, num_filters=64, filter_size=(3, 3),
                                             nonlinearity=lasagne.nonlinearities.elu,
                                             pad="same", stride=1))

        network = L.Upscale2DLayer(network, 2, mode='repeat')
        network = L.batch_norm(L.Conv2DLayer(network, num_filters=64, filter_size=(3, 3),
                                             nonlinearity=lasagne.nonlinearities.elu,
                                             pad="same", stride=1))
        network = L.batch_norm(L.Conv2DLayer(network, num_filters=64, filter_size=(3, 3),
                                             nonlinearity=lasagne.nonlinearities.elu,
                                             pad="same", stride=1))

        network = L.Upscale2DLayer(network, 2, mode='repeat')
        network = L.batch_norm(L.Conv2DLayer(network, num_filters=64, filter_size=(3, 3),
                                             nonlinearity=lasagne.nonlinearities.elu,
                                             pad="same", stride=1))
        network = L.batch_norm(L.Conv2DLayer(network, num_filters=64, filter_size=(3, 3),
                                             nonlinearity=lasagne.nonlinearities.elu,
                                             pad="same", stride=1))

        network = L.Conv2DLayer(network, num_filters=3, filter_size=(3, 3),
                                nonlinearity=lasagne.nonlinearities.leaky_rectify,
                                pad="same", stride=1)


        return network

    def build_gen2(self, input_gen, batch_size):
        network = L.InputLayer(shape=(batch_size, self.noise_dim), input_var=input_gen)

        network = L.DenseLayer(network, num_units=4096,
                               nonlinearity=lasagne.nonlinearities.identity)

        network = L.ReshapeLayer(network, shape=(batch_size, 64, 8, 8))

        network = L.batch_norm(L.Conv2DLayer(network, num_filters=64, filter_size=(3, 3),
                                             nonlinearity=lasagne.nonlinearities.elu,
                                             pad="same", stride=1))
        network = L.batch_norm(L.Conv2DLayer(network, num_filters=64, filter_size=(3, 3),
                                             nonlinearity=lasagne.nonlinearities.elu,
                                             pad="same", stride=1))
        network = L.batch_norm(L.Conv2DLayer(network, num_filters=64, filter_size=(3, 3),
                                             nonlinearity=lasagne.nonlinearities.elu,
                                             pad="same", stride=1))

        network = L.Upscale2DLayer(network, 2, mode='repeat')
        network = L.batch_norm(L.Conv2DLayer(network, num_filters=64, filter_size=(3, 3),
                                             nonlinearity=lasagne.nonlinearities.elu,
                                             pad="same", stride=1))
        network = L.batch_norm(L.Conv2DLayer(network, num_filters=64, filter_size=(3, 3),
                                             nonlinearity=lasagne.nonlinearities.elu,
                                             pad="same", stride=1))
        network = L.batch_norm(L.Conv2DLayer(network, num_filters=64, filter_size=(3, 3),
                                             nonlinearity=lasagne.nonlinearities.elu,
                                             pad="same", stride=1))

        network = L.Upscale2DLayer(network, 2, mode='repeat')
        network = L.batch_norm(L.Conv2DLayer(network, num_filters=64, filter_size=(3, 3),
                                             nonlinearity=lasagne.nonlinearities.elu,
                                             pad="same", stride=1))
        network = L.batch_norm(L.Conv2DLayer(network, num_filters=64, filter_size=(3, 3),
                                             nonlinearity=lasagne.nonlinearities.elu,
                                             pad="same", stride=1))
        network = L.batch_norm(L.Conv2DLayer(network, num_filters=64, filter_size=(3, 3),
                                             nonlinearity=lasagne.nonlinearities.elu,
                                             pad="same", stride=1))

        network = L.Conv2DLayer(network, num_filters=3, filter_size=(3, 3),
                                nonlinearity=lasagne.nonlinearities.leaky_rectify,
                                pad="same", stride=1)

        # output is 32 x 32
        return network

    def build_dis2(self, input_dis, batch_size):
        network = L.InputLayer(shape=(batch_size, 3, 32, 32), input_var=input_dis)

        network = L.batch_norm(L.Conv2DLayer(network, num_filters=64, filter_size=(3, 3),
                                             nonlinearity=lasagne.nonlinearities.elu,
                                             pad="same", stride=1))
        network = L.batch_norm(L.Conv2DLayer(network, num_filters=64, filter_size=(3, 3),
                                             nonlinearity=lasagne.nonlinearities.elu,
                                             pad="same", stride=1))
        network = L.batch_norm(L.Conv2DLayer(network, num_filters=64, filter_size=(3, 3),
                                             nonlinearity=lasagne.nonlinearities.elu,
                                             pad="same", stride=1))

        network = L.batch_norm(L.Conv2DLayer(network, num_filters=128, filter_size=(3, 3),
                                             nonlinearity=lasagne.nonlinearities.elu,
                                             pad=1, stride=2))
        network = L.batch_norm(L.Conv2DLayer(network, num_filters=128, filter_size=(3, 3),
                                             nonlinearity=lasagne.nonlinearities.elu,
                                             pad="same", stride=1))
        network = L.batch_norm(L.Conv2DLayer(network, num_filters=128, filter_size=(3, 3),
                                             nonlinearity=lasagne.nonlinearities.elu,
                                             pad="same"))
        network = L.batch_norm(L.Conv2DLayer(network, num_filters=128, filter_size=(3, 3),
                                             nonlinearity=lasagne.nonlinearities.elu,
                                             pad="same"))

        network = L.batch_norm(L.Conv2DLayer(network, num_filters=192, filter_size=(3, 3),
                                             nonlinearity=lasagne.nonlinearities.elu,
                                             pad=1, stride=2))
        network = L.batch_norm(L.Conv2DLayer(network, num_filters=192, filter_size=(3, 3),
                                             nonlinearity=lasagne.nonlinearities.elu,
                                             pad="same", stride=1))
        network = L.batch_norm(L.Conv2DLayer(network, num_filters=192, filter_size=(3, 3),
                                             nonlinearity=lasagne.nonlinearities.elu,
                                             pad="same"))
        network = L.batch_norm(L.Conv2DLayer(network, num_filters=192, filter_size=(3, 3),
                                             nonlinearity=lasagne.nonlinearities.elu,
                                             pad="same"))

        # 8x8 image size here
        network = L.DenseLayer(network, num_units=128,
                               nonlinearity=lasagne.nonlinearities.identity)

        network = L.DenseLayer(network, num_units=4096,
                               nonlinearity=lasagne.nonlinearities.elu)

        network = L.ReshapeLayer(network, shape=(batch_size, 64, 8, 8))

        network = L.batch_norm(L.Conv2DLayer(network, num_filters=64, filter_size=(3, 3),
                                             nonlinearity=lasagne.nonlinearities.elu,
                                             pad="same", stride=1))
        network = L.batch_norm(L.Conv2DLayer(network, num_filters=64, filter_size=(3, 3),
                                             nonlinearity=lasagne.nonlinearities.elu,
                                             pad="same", stride=1))
        network = L.batch_norm(L.Conv2DLayer(network, num_filters=64, filter_size=(3, 3),
                                             nonlinearity=lasagne.nonlinearities.elu,
                                             pad="same", stride=1))

        network = L.Upscale2DLayer(network, 2, mode='repeat')
        network = L.batch_norm(L.Conv2DLayer(network, num_filters=64, filter_size=(3, 3),
                                             nonlinearity=lasagne.nonlinearities.elu,
                                             pad="same", stride=1))
        network = L.batch_norm(L.Conv2DLayer(network, num_filters=64, filter_size=(3, 3),
                                             nonlinearity=lasagne.nonlinearities.elu,
                                             pad="same", stride=1))
        network = L.batch_norm(L.Conv2DLayer(network, num_filters=64, filter_size=(3, 3),
                                             nonlinearity=lasagne.nonlinearities.elu,
                                             pad="same", stride=1))

        network = L.Upscale2DLayer(network, 2, mode='repeat')
        network = L.batch_norm(L.Conv2DLayer(network, num_filters=64, filter_size=(3, 3),
                                             nonlinearity=lasagne.nonlinearities.elu,
                                             pad="same", stride=1))
        network = L.batch_norm(L.Conv2DLayer(network, num_filters=64, filter_size=(3, 3),
                                             nonlinearity=lasagne.nonlinearities.elu,
                                             pad="same", stride=1))
        network = L.batch_norm(L.Conv2DLayer(network, num_filters=64, filter_size=(3, 3),
                                             nonlinearity=lasagne.nonlinearities.elu,
                                             pad="same", stride=1))

        network = L.Conv2DLayer(network, num_filters=3, filter_size=(3, 3),
                                nonlinearity=lasagne.nonlinearities.leaky_rectify,
                                pad="same", stride=1)

        return network