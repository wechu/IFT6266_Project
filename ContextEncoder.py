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



class BEGAN():
    def __init__(self, name="", gamma=0.5, k_initial=0, loss_adv=0.5, dis_type="center_to_center", use_caption=False):
        self.name = name
        self.gen = None  # generator
        self.dis = None  # discriminator
        self.gamma = gamma
        self.k = k_initial
        self.loss_adv = loss_adv
        self.dis_type = dis_type  # center_to_center, full_to_full, full_to_center
        self.use_caption = use_caption
        self.updates = None
        pass

    def compile_theano(self, learn_rate, batch_size, mode, momentum=0.9, training=True,
                       inputs_shape=None, targets_shape=None, macro_batch_size=None):
        border_var = T.tensor4('input', dtype='float32')
        # noise_var = T.matrix('noise', dtype='float32')  # no noise
        target_var = T.tensor4('target', dtype='float32')
        if self.use_caption:
            caption_var = T.matrix('caption', dtype='float32')

        if self.dis_type in ("full_to_full", "full_to_center"):
            center = (32, 32)  # center of image
            # reconstruct the original image
            real_full = T.set_subtensor(border_var[:, :, center[0]-16:center[0]+16, center[1]-16:center[1]+16], target_var)

        # initialize model
        if mode == "v1":
            self.gen = self.build_gen(border_var, batch_size)
            self.dis = self.build_dis(target_var, batch_size)  # discriminator is actually an autoencoder
        if mode == "v2":
            self.gen = self.build_gen2(border_var, batch_size)
            self.dis = self.build_dis2(real_full, batch_size)
        if mode == "v3":
            enc = self.build_enc3(None, batch_size) # share encoder for both dis and gen
            self.gen = self.build_gen3(enc, batch_size)
            self.dis = self.build_dis3(enc, batch_size)
        if mode == "v4":
            self.gen = self.build_net4(border_var, batch_size)
            self.dis = self.build_net4(real_full, batch_size)
        if mode == "v5":
            self.gen = self.build_net5(border_var, batch_size)
            self.dis = self.build_net5(real_full, batch_size)
        if mode == "v6":
            self.gen = self.build_gen6(border_var, batch_size)
            self.dis = self.build_dis6(real_full, batch_size)
        if mode == "v7":
            self.gen = self.build_gen7(border_var, batch_size)
            self.dis = self.build_dis7(target_var, batch_size)
        if mode == "v8":
            self.gen = self.build_gen8(border_var, caption_var, batch_size)
            self.dis = self.build_dis8(target_var, caption_var, batch_size)

        # get predictions from discriminator (autoencoder)
        if self.dis_type in ("center_to_center"):  # use only the center in the dis
            real_autoenc = L.get_output(self.dis)  # inputs real center patch to dis
            fake_center = L.get_output(self.gen)  # inputs border
            if self.use_caption:
                fake_autoenc = L.get_output(self.dis, inputs={target_var: fake_center, caption_var: caption_var})
            else:
                fake_autoenc = L.get_output(self.dis, inputs=fake_center)

        else:  # use the full image
            real_autoenc = L.get_output(self.dis, inputs=real_full)
            fake_center = L.get_output(self.gen, inputs=border_var)
            fake_full = T.set_subtensor(border_var[:, :, center[0]-16:center[0]+16, center[1]-16:center[1]+16], fake_center)
            fake_autoenc = L.get_output(self.dis, inputs=fake_full)


        # use L1 reconstruction loss
        k = theano.shared(np.array(self.k, dtype='float32'))    # balancing variable

        if self.dis_type == "center_to_center":
            loss_real = T.abs_(real_autoenc - target_var)
            loss_fake = T.abs_(fake_autoenc - fake_center)
        elif self.dis_type == "full_to_center":
            # output only the center for both dis and gen
            loss_real = T.abs_(real_autoenc - target_var)
            loss_fake = T.abs_(fake_autoenc - fake_center)
        elif self.dis_type == "full_to_full":
            loss_real = T.abs_(real_autoenc - real_full)
            loss_fake = T.abs_(fake_autoenc - fake_full)
        else:
            loss_real = None
            loss_fake = None

        loss_dis = T.mean(loss_real - k * loss_fake)

        # new loss: make generator also respect the actual center image (Context encoder)
        loss_gen = self.loss_adv * T.mean(loss_fake) + (1-self.loss_adv) * T.sqrt(T.mean(lasagne.objectives.squared_error(fake_center, target_var)))
        # loss_gen = T.mean(loss_fake)

        # get updates
        lr = theano.shared(np.array(learn_rate, dtype='float32'))
        self.lr = lr

        params_gen = L.get_all_params(self.gen, trainable=True)
        params_dis = L.get_all_params(self.dis, trainable=True)

        updates = lasagne.updates.adam(loss_gen, params_gen, learning_rate=lr, beta1=momentum)
        if mode in ("v1", "v2", "v3", "v4"):  # oops, mistake from before. Keep it for backwards compatibility
            updates.update(lasagne.updates.momentum(loss_dis, params_dis, learning_rate=lr))
        else:
            updates.update(lasagne.updates.adam(loss_dis, params_dis, learning_rate=lr, beta1=momentum))
        updates.update({k: k + 0.01 * T.mean(self.gamma * loss_real - loss_fake)})
        self.updates = updates

        print("Compiling methods...")
        if not self.use_caption:
            # compile methods to train (trains everything at once)
            if training:
                self.macro_batch_inputs = theano.shared(np.empty((macro_batch_size,) + inputs_shape, dtype='float32'), borrow=True)
                self.macro_batch_targets = theano.shared(np.empty((macro_batch_size,) + targets_shape, dtype='float32'), borrow=True)
                index = T.lscalar()
                self.train_method = theano.function([index], [T.mean(loss_real), T.mean(loss_fake), k, T.mean(loss_gen)],
                                                    updates=updates,
                                                    givens={border_var: self.macro_batch_inputs[index*batch_size:(index+1)*batch_size],
                                                            target_var: self.macro_batch_targets[index*batch_size:(index+1)*batch_size]})

            # method to generate just images
            self.gen_method = theano.function([border_var], L.get_output(self.gen, inputs=border_var, deterministic=True))

        if self.use_caption: # we input the sentence embedding in addition to the border
            if training:
                self.macro_batch_inputs = theano.shared(np.empty((macro_batch_size,) + inputs_shape, dtype='float32'),
                                                        borrow=True)
                self.macro_batch_targets = theano.shared(np.empty((macro_batch_size,) + targets_shape, dtype='float32'),
                                                         borrow=True)
                self.macro_batch_captions = theano.shared(np.empty((macro_batch_size,) + (4800,), dtype='float32'),
                                                         borrow=True)
                index = T.lscalar()
                self.train_method = theano.function([index], [T.mean(loss_real), T.mean(loss_fake), k, T.mean(loss_gen)],
                                                    updates=updates,
                                                    givens={border_var: self.macro_batch_inputs[
                                                                        index * batch_size:(index + 1) * batch_size],
                                                            target_var: self.macro_batch_targets[
                                                                        index * batch_size:(index + 1) * batch_size],
                                                            caption_var: self.macro_batch_captions[
                                                                         index * batch_size:(index + 1) * batch_size]
                                                            })

            # method to generate just images
            self.gen_method = theano.function([border_var, caption_var],
                                              L.get_output(self.gen, inputs={border_var:border_var, caption_var:caption_var}, deterministic=True))
        return

    def compute_global_loss(self, loss_real, loss_fake):
        ''' Computes global loss to assess training '''
        return loss_real + abs(self.gamma * loss_real - loss_fake)

    def train(self, inputs, targets, captions=None, batch_size=32, macro_batch_size=8192, start_iter=1, log_file=None, anneal=False):
        save_freq = 2000
        x_train_num = targets.shape[0]
        gen_file = self.name + "_gen.txt"
        dis_file = self.name + "_dis.txt"
        check_file = self.name + "_check.txt"
        gen_string = ""
        dis_string = ""
        check_string = ""
        log_string = ""
        if captions is not None:
            captions = np.array(captions).astype('float32')

        # training loop
        num_epochs = 100
        iter = start_iter
        index_list = np.arange(x_train_num, dtype='int32')  # used for shuffling

        print("Start training...")
        time_start = time.time()
        for epoch in range(num_epochs):
            log_string += "----- Epoch" + str(epoch) + " " + str((time.time() - time_start)/60) + "mins -----\n"
            time_start = time.time()  # reset the timer for each epoch

            # Shuffle all examples every epoch
            np.random.shuffle(index_list)

            for macro_batch_i in range(x_train_num // macro_batch_size):
                # Set the current macro batch
                shuffle_ind = index_list[macro_batch_i*macro_batch_size: (macro_batch_i+1)*macro_batch_size]
                self.macro_batch_inputs.set_value(inputs[shuffle_ind], borrow=True)
                self.macro_batch_targets.set_value(targets[shuffle_ind], borrow=True)
                if self.use_caption:
                    self.macro_batch_captions.set_value(captions[shuffle_ind, epoch % 5], borrow=True) # cycle through the 5 captions every epoch

                for i in range(macro_batch_size // batch_size):
                    # train on one minibatch
                    loss_real, loss_fake, k, loss_gen = self.train_method(i)  # index of minibatch is passed

                    log_string += str(epoch) + " " + str(i) + "\tk: {:.3f} Check:{:.6f} Gen:{:.6f} Dis:{:.6f} Real{:.6f} Fake{:.6f}\n".format(
                            float(np.array(k)), float(self.compute_global_loss(loss_real, loss_fake)), float(loss_gen), float(loss_real - k * loss_fake),
                            float(loss_real), float(loss_fake))

                    gen_string += str(float(loss_gen)) + "\n"
                    dis_string += str(float(loss_real - k * loss_fake)) + "\n"
                    check_string += str(self.compute_global_loss(loss_real, loss_fake)) + "\n"

                    # Validation set and save weights
                    if iter % save_freq == 0 or iter == 1:
                        utils.save_model_weights(self.gen,
                                                 "{}_gen_weights_it_{}.pkl".format(self.name, iter))
                        # utils.save_model_weights(self.dis,
                        #                          "{}_dis_weights_it_{}.pkl".format(self.name, iter))
                        utils.save_updates(self.updates, "{}_updates_it_{}.pkl".format(self.name, iter))
                        # Write results to file and reset strings
                        with open(gen_file, 'a') as f:
                            f.write(gen_string)
                        with open(dis_file, 'a') as f:
                            f.write(dis_string)
                        with open(check_file, 'a') as f:
                            f.write(check_string)
                        with open(log_file, 'a') as f:
                            f.write(log_string)
                        gen_string = ""
                        dis_string = ""
                        check_string = ""
                        log_string = ""

                    ##reduce learning rate
                    if anneal and iter % 10000 == 0 and iter > 1:
                        self.lr.set_value(self.lr.get_value() * np.array(0.7).astype('float32'))
                        log_string += "Reduced learning rate: {}\n".format((float(np.array(self.lr.get_value()))))

                    iter += 1

        print("Total time", time.time() - time_start)
        # test a few values

        return



    # without full dis
    def build_gen(self, input_gen, batch_size):
        # we input the border (no noise)
        network = L.InputLayer(shape=(batch_size, 3, 64, 64), input_var=input_gen)

        network = L.batch_norm(L.Conv2DLayer(network, num_filters=64, filter_size=(3, 3),
                                             nonlinearity=lasagne.nonlinearities.elu,
                                             pad="same", stride=1))
        network = L.batch_norm(L.Conv2DLayer(network, num_filters=64, filter_size=(3, 3),
                                             nonlinearity=lasagne.nonlinearities.elu,
                                             pad="same", stride=1))

        network = L.batch_norm(L.Conv2DLayer(network, num_filters=64, filter_size=(3, 3),
                                             nonlinearity=lasagne.nonlinearities.elu,
                                             pad=1, stride=2))
        network = L.batch_norm(L.Conv2DLayer(network, num_filters=64, filter_size=(3, 3),
                                             nonlinearity=lasagne.nonlinearities.elu,
                                             pad="same", stride=1))
        network = L.batch_norm(L.Conv2DLayer(network, num_filters=64, filter_size=(3, 3),
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

        network = L.batch_norm(L.Conv2DLayer(network, num_filters=128, filter_size=(3, 3),
                                             nonlinearity=lasagne.nonlinearities.elu,
                                             pad=1, stride=2))
        network = L.batch_norm(L.Conv2DLayer(network, num_filters=128, filter_size=(3, 3),
                                             nonlinearity=lasagne.nonlinearities.elu,
                                             pad="same", stride=1))
        network = L.batch_norm(L.Conv2DLayer(network, num_filters=128, filter_size=(3, 3),
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
        # We input only the 32x32 center patch
        network = L.InputLayer(shape=(batch_size, 3, 32, 32), input_var=input_dis)

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

        network = L.batch_norm(L.Conv2DLayer(network, num_filters=192, filter_size=(3, 3),
                                             nonlinearity=lasagne.nonlinearities.elu,
                                             pad=1, stride=2))
        network = L.batch_norm(L.Conv2DLayer(network, num_filters=192, filter_size=(3, 3),
                                             nonlinearity=lasagne.nonlinearities.elu,
                                             pad="same", stride=1))
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

    # with full dis
    def build_gen2(self, input_gen, batch_size):
        # we input the border (no noise)
        network = L.InputLayer(shape=(batch_size, 3, 64, 64), input_var=input_gen)

        network = L.batch_norm(L.Conv2DLayer(network, num_filters=32, filter_size=(3, 3),
                                             nonlinearity=lasagne.nonlinearities.elu,
                                             pad="same", stride=1))
        network = L.batch_norm(L.Conv2DLayer(network, num_filters=32, filter_size=(3, 3),
                                             nonlinearity=lasagne.nonlinearities.elu,
                                             pad="same", stride=1))
        network = L.batch_norm(L.Conv2DLayer(network, num_filters=32, filter_size=(3, 3),
                                             nonlinearity=lasagne.nonlinearities.elu,
                                             pad=1, stride=2))

        network = L.batch_norm(L.Conv2DLayer(network, num_filters=64, filter_size=(3, 3),
                                             nonlinearity=lasagne.nonlinearities.elu,
                                             pad="same", stride=1))
        network = L.batch_norm(L.Conv2DLayer(network, num_filters=64, filter_size=(3, 3),
                                             nonlinearity=lasagne.nonlinearities.elu,
                                             pad="same"))
        network = L.batch_norm(L.Conv2DLayer(network, num_filters=64, filter_size=(3, 3),
                                             nonlinearity=lasagne.nonlinearities.elu,
                                             pad=1, stride=2))

        network = L.batch_norm(L.Conv2DLayer(network, num_filters=96, filter_size=(3, 3),
                                             nonlinearity=lasagne.nonlinearities.elu,
                                             pad="same", stride=1))
        network = L.batch_norm(L.Conv2DLayer(network, num_filters=96, filter_size=(3, 3),
                                             nonlinearity=lasagne.nonlinearities.elu,
                                             pad="same"))
        network = L.batch_norm(L.Conv2DLayer(network, num_filters=96, filter_size=(3, 3),
                                             nonlinearity=lasagne.nonlinearities.elu,
                                             pad=1, stride=2))

        network = L.batch_norm(L.Conv2DLayer(network, num_filters=128, filter_size=(3, 3),
                                             nonlinearity=lasagne.nonlinearities.elu,
                                             pad="same", stride=1))
        network = L.batch_norm(L.Conv2DLayer(network, num_filters=128, filter_size=(3, 3),
                                             nonlinearity=lasagne.nonlinearities.elu,
                                             pad="same", stride=1))
        # 8x8 image size here
        network = L.DenseLayer(network, num_units=128,
                               nonlinearity=lasagne.nonlinearities.identity)

        network = L.DenseLayer(network, num_units=2048,
                               nonlinearity=lasagne.nonlinearities.elu)

        network = L.ReshapeLayer(network, shape=(batch_size, 32, 8, 8))

        network = L.batch_norm(L.Conv2DLayer(network, num_filters=32, filter_size=(3, 3),
                                             nonlinearity=lasagne.nonlinearities.elu,
                                             pad="same", stride=1))
        network = L.batch_norm(L.Conv2DLayer(network, num_filters=32, filter_size=(3, 3),
                                             nonlinearity=lasagne.nonlinearities.elu,
                                             pad="same", stride=1))

        network = L.Upscale2DLayer(network, 2, mode='repeat')
        network = L.batch_norm(L.Conv2DLayer(network, num_filters=32, filter_size=(3, 3),
                                             nonlinearity=lasagne.nonlinearities.elu,
                                             pad="same", stride=1))
        network = L.batch_norm(L.Conv2DLayer(network, num_filters=32, filter_size=(3, 3),
                                             nonlinearity=lasagne.nonlinearities.elu,
                                             pad="same", stride=1))

        network = L.Upscale2DLayer(network, 2, mode='repeat')
        network = L.batch_norm(L.Conv2DLayer(network, num_filters=32, filter_size=(3, 3),
                                             nonlinearity=lasagne.nonlinearities.elu,
                                             pad="same", stride=1))
        network = L.batch_norm(L.Conv2DLayer(network, num_filters=32, filter_size=(3, 3),
                                             nonlinearity=lasagne.nonlinearities.elu,
                                             pad="same", stride=1))


        network = L.Conv2DLayer(network, num_filters=3, filter_size=(3, 3),
                                nonlinearity=lasagne.nonlinearities.leaky_rectify,
                                pad="same", stride=1)

        # output is 32 x 32
        return network

    def build_dis2(self, input_dis, batch_size):
        # We input the whole 64x64 image
        network = L.InputLayer(shape=(batch_size, 3, 64, 64), input_var=input_dis)
        network = L.batch_norm(L.Conv2DLayer(network, num_filters=32, filter_size=(3, 3),
                                             nonlinearity=lasagne.nonlinearities.elu,
                                             pad="same", stride=1))
        network = L.batch_norm(L.Conv2DLayer(network, num_filters=32, filter_size=(3, 3),
                                             nonlinearity=lasagne.nonlinearities.elu,
                                             pad="same", stride=1))
        network = L.batch_norm(L.Conv2DLayer(network, num_filters=32, filter_size=(3, 3),
                                             nonlinearity=lasagne.nonlinearities.elu,
                                             pad=1, stride=2))

        network = L.batch_norm(L.Conv2DLayer(network, num_filters=64, filter_size=(3, 3),
                                             nonlinearity=lasagne.nonlinearities.elu,
                                             pad="same", stride=1))
        network = L.batch_norm(L.Conv2DLayer(network, num_filters=64, filter_size=(3, 3),
                                             nonlinearity=lasagne.nonlinearities.elu,
                                             pad="same", stride=1))
        network = L.batch_norm(L.Conv2DLayer(network, num_filters=64, filter_size=(3, 3),
                                             nonlinearity=lasagne.nonlinearities.elu,
                                             pad=1, stride=2))

        network = L.batch_norm(L.Conv2DLayer(network, num_filters=96, filter_size=(3, 3),
                                             nonlinearity=lasagne.nonlinearities.elu,
                                             pad="same", stride=1))
        network = L.batch_norm(L.Conv2DLayer(network, num_filters=96, filter_size=(3, 3),
                                             nonlinearity=lasagne.nonlinearities.elu,
                                             pad="same"))
        network = L.batch_norm(L.Conv2DLayer(network, num_filters=96, filter_size=(3, 3),
                                             nonlinearity=lasagne.nonlinearities.elu,
                                             pad=1, stride=2))

        network = L.batch_norm(L.Conv2DLayer(network, num_filters=128, filter_size=(3, 3),
                                             nonlinearity=lasagne.nonlinearities.elu,
                                             pad="same", stride=1))
        network = L.batch_norm(L.Conv2DLayer(network, num_filters=128, filter_size=(3, 3),
                                             nonlinearity=lasagne.nonlinearities.elu,
                                             pad="same", stride=1))

        # 8x8 image size here
        network = L.DenseLayer(network, num_units=128,
                               nonlinearity=lasagne.nonlinearities.identity)

        network = L.DenseLayer(network, num_units=2048,
                               nonlinearity=lasagne.nonlinearities.elu)

        network = L.ReshapeLayer(network, shape=(batch_size, 32, 8, 8))

        network = L.batch_norm(L.Conv2DLayer(network, num_filters=32, filter_size=(3, 3),
                                             nonlinearity=lasagne.nonlinearities.elu,
                                             pad="same", stride=1))
        network = L.batch_norm(L.Conv2DLayer(network, num_filters=32, filter_size=(3, 3),
                                             nonlinearity=lasagne.nonlinearities.elu,
                                             pad="same", stride=1))

        network = L.Upscale2DLayer(network, 2, mode='repeat')
        network = L.batch_norm(L.Conv2DLayer(network, num_filters=32, filter_size=(3, 3),
                                             nonlinearity=lasagne.nonlinearities.elu,
                                             pad="same", stride=1))
        network = L.batch_norm(L.Conv2DLayer(network, num_filters=32, filter_size=(3, 3),
                                             nonlinearity=lasagne.nonlinearities.elu,
                                             pad="same", stride=1))

        network = L.Upscale2DLayer(network, 2, mode='repeat')
        network = L.batch_norm(L.Conv2DLayer(network, num_filters=32, filter_size=(3, 3),
                                             nonlinearity=lasagne.nonlinearities.elu,
                                             pad="same", stride=1))
        network = L.batch_norm(L.Conv2DLayer(network, num_filters=32, filter_size=(3, 3),
                                             nonlinearity=lasagne.nonlinearities.elu,
                                             pad="same", stride=1))

        network = L.Upscale2DLayer(network, 2, mode='repeat')
        network = L.batch_norm(L.Conv2DLayer(network, num_filters=32, filter_size=(3, 3),
                                             nonlinearity=lasagne.nonlinearities.elu,
                                             pad="same", stride=1))
        network = L.batch_norm(L.Conv2DLayer(network, num_filters=32, filter_size=(3, 3),
                                             nonlinearity=lasagne.nonlinearities.elu,
                                             pad="same", stride=1))

        network = L.Conv2DLayer(network, num_filters=3, filter_size=(3, 3),
                                nonlinearity=lasagne.nonlinearities.leaky_rectify,
                                pad="same", stride=1)

        return network

    # try sharing the encoder part for both generator and discriminator
    def build_enc3(self, input, batch_size):
        # we input the border (no noise)
        network = L.InputLayer(shape=(batch_size, 3, 64, 64), input_var=input)

        network = (L.Conv2DLayer(network, num_filters=48, filter_size=(5, 5),
                                             nonlinearity=lasagne.nonlinearities.elu,
                                             pad="same", stride=1))
        network = (L.Conv2DLayer(network, num_filters=48, filter_size=(5, 5),
                                             nonlinearity=lasagne.nonlinearities.elu,
                                             pad=2, stride=2))

        network = (L.Conv2DLayer(network, num_filters=96, filter_size=(3, 3),
                                             nonlinearity=lasagne.nonlinearities.elu,
                                             pad="same", stride=1))
        network = (L.Conv2DLayer(network, num_filters=96, filter_size=(3, 3),
                                             nonlinearity=lasagne.nonlinearities.elu,
                                             pad="same"))
        network = (L.Conv2DLayer(network, num_filters=96, filter_size=(3, 3),
                                             nonlinearity=lasagne.nonlinearities.elu,
                                             pad=1, stride=2))

        network = (L.Conv2DLayer(network, num_filters=128, filter_size=(3, 3),
                                             nonlinearity=lasagne.nonlinearities.elu,
                                             pad="same", stride=1))
        network = (L.Conv2DLayer(network, num_filters=128, filter_size=(3, 3),
                                             nonlinearity=lasagne.nonlinearities.elu,
                                             pad="same"))

        network = (L.Conv2DLayer(network, num_filters=32, filter_size=(1, 1),
                                             nonlinearity=lasagne.nonlinearities.elu,
                                             pad="same"))
        # 16x16 image size here
        network = L.batch_norm(L.DenseLayer(network, num_units=512,
                               nonlinearity=lasagne.nonlinearities.very_leaky_rectify))

        return network

    def build_gen3(self, network, batch_size):
        # we input the hidden rep
        network = L.DenseLayer(network, num_units=8192,
                               nonlinearity=lasagne.nonlinearities.elu)

        network = L.ReshapeLayer(network, shape=(batch_size, 32, 16, 16))

        network = (L.Conv2DLayer(network, num_filters=64, filter_size=(3, 3),
                                             nonlinearity=lasagne.nonlinearities.elu,
                                             pad="same", stride=1))
        network = (L.Conv2DLayer(network, num_filters=64, filter_size=(3, 3),
                                             nonlinearity=lasagne.nonlinearities.elu,
                                             pad="same", stride=1))


        network = L.Upscale2DLayer(network, 2, mode='repeat')
        network = (L.Conv2DLayer(network, num_filters=48, filter_size=(3, 3),
                                             nonlinearity=lasagne.nonlinearities.elu,
                                             pad="same", stride=1))
        network = (L.Conv2DLayer(network, num_filters=48, filter_size=(3, 3),
                                             nonlinearity=lasagne.nonlinearities.elu,
                                             pad="same", stride=1))

        network = L.LocallyConnected2DLayer(network, num_filters=3, filter_size=(1, 1),
                                            nonlinearity=lasagne.nonlinearities.LeakyRectify(0.01),
                                            pad="same")


        # output is 32 x 32
        return network

    def build_dis3(self, network, batch_size):
        # We input the hidden rep
        network = L.DenseLayer(network, num_units=8192,
                               nonlinearity=lasagne.nonlinearities.elu)

        network = L.ReshapeLayer(network, shape=(batch_size, 32, 16, 16))

        network = (L.Conv2DLayer(network, num_filters=64, filter_size=(3, 3),
                                             nonlinearity=lasagne.nonlinearities.elu,
                                             pad="same", stride=1))
        network = (L.Conv2DLayer(network, num_filters=64, filter_size=(3, 3),
                                             nonlinearity=lasagne.nonlinearities.elu,
                                             pad="same", stride=1))

        network = L.Upscale2DLayer(network, 2, mode='repeat')
        network = (L.Conv2DLayer(network, num_filters=48, filter_size=(3, 3),
                                             nonlinearity=lasagne.nonlinearities.elu,
                                             pad="same", stride=1))
        network = (L.Conv2DLayer(network, num_filters=48, filter_size=(3, 3),
                                             nonlinearity=lasagne.nonlinearities.elu,
                                             pad="same", stride=1))

        network = L.Upscale2DLayer(network, 2, mode='repeat')
        network = (L.Conv2DLayer(network, num_filters=32, filter_size=(5, 5),
                                             nonlinearity=lasagne.nonlinearities.elu,
                                             pad="same", stride=1))
        network = (L.Conv2DLayer(network, num_filters=32, filter_size=(5, 5),
                                             nonlinearity=lasagne.nonlinearities.elu,
                                             pad="same", stride=1))

        network = L.LocallyConnected2DLayer(network, num_filters=3, filter_size=(1, 1),
                                            nonlinearity=lasagne.nonlinearities.LeakyRectify(0.01),
                                            pad="same")
        return network

    # try dis takes 64x64 image as input and reconstructs 32x32
    # gen and dis are identical

    def build_net4(self, input, batch_size):
        # we input the border (no noise)
        network = L.InputLayer(shape=(batch_size, 3, 64, 64), input_var=input)

        network = L.batch_norm(L.Conv2DLayer(network, num_filters=32, filter_size=(3, 3),
                                             nonlinearity=lasagne.nonlinearities.elu,
                                             pad="same", stride=1))
        network = L.batch_norm(L.Conv2DLayer(network, num_filters=32, filter_size=(3, 3),
                                             nonlinearity=lasagne.nonlinearities.elu,
                                             pad="same", stride=1))
        network = L.batch_norm(L.Conv2DLayer(network, num_filters=32, filter_size=(3, 3),
                                             nonlinearity=lasagne.nonlinearities.elu,
                                             pad=1, stride=2))

        network = L.batch_norm(L.Conv2DLayer(network, num_filters=64, filter_size=(3, 3),
                                             nonlinearity=lasagne.nonlinearities.elu,
                                             pad="same", stride=1))
        network = L.batch_norm(L.Conv2DLayer(network, num_filters=64, filter_size=(3, 3),
                                             nonlinearity=lasagne.nonlinearities.elu,
                                             pad="same"))
        network = L.batch_norm(L.Conv2DLayer(network, num_filters=64, filter_size=(3, 3),
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
                                             pad=1, stride=2))

        network = L.batch_norm(L.Conv2DLayer(network, num_filters=128, filter_size=(3, 3),
                                             nonlinearity=lasagne.nonlinearities.elu,
                                             pad="same", stride=1))
        network = L.batch_norm(L.Conv2DLayer(network, num_filters=128, filter_size=(3, 3),
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

    # Similar to net4 without batch norm and more filters
    def build_net5(self, input, batch_size):
        # we input the border (no noise)
        network = L.InputLayer(shape=(batch_size, 3, 64, 64), input_var=input)

        network = (L.Conv2DLayer(network, num_filters=48, filter_size=(5, 5),
                                             nonlinearity=lasagne.nonlinearities.elu,
                                             pad="same", stride=1))
        network = (L.Conv2DLayer(network, num_filters=48, filter_size=(3, 3),
                                             nonlinearity=lasagne.nonlinearities.elu,
                                             pad="same", stride=1))
        network = (L.Conv2DLayer(network, num_filters=48, filter_size=(3, 3),
                                             nonlinearity=lasagne.nonlinearities.elu,
                                             pad=1, stride=2))

        network = (L.Conv2DLayer(network, num_filters=64, filter_size=(3, 3),
                                             nonlinearity=lasagne.nonlinearities.elu,
                                             pad="same", stride=1))
        network = (L.Conv2DLayer(network, num_filters=64, filter_size=(3, 3),
                                             nonlinearity=lasagne.nonlinearities.elu,
                                             pad="same"))
        network = (L.Conv2DLayer(network, num_filters=64, filter_size=(3, 3),
                                             nonlinearity=lasagne.nonlinearities.elu,
                                             pad=1, stride=2))

        network = (L.Conv2DLayer(network, num_filters=128, filter_size=(3, 3),
                                             nonlinearity=lasagne.nonlinearities.elu,
                                             pad="same", stride=1))
        network = (L.Conv2DLayer(network, num_filters=128, filter_size=(3, 3),
                                             nonlinearity=lasagne.nonlinearities.elu,
                                             pad="same"))
        network = (L.Conv2DLayer(network, num_filters=128, filter_size=(3, 3),
                                             nonlinearity=lasagne.nonlinearities.elu,
                                             pad=1, stride=2))

        network = (L.Conv2DLayer(network, num_filters=192, filter_size=(3, 3),
                                             nonlinearity=lasagne.nonlinearities.elu,
                                             pad="same", stride=1))
        network = (L.Conv2DLayer(network, num_filters=192, filter_size=(3, 3),
                                             nonlinearity=lasagne.nonlinearities.elu,
                                             pad="same"))
        # 8x8 image size here
        network = L.DenseLayer(network, num_units=128,
                               nonlinearity=lasagne.nonlinearities.identity)

        network = L.DenseLayer(network, num_units=4096,
                               nonlinearity=lasagne.nonlinearities.elu)

        network = L.ReshapeLayer(network, shape=(batch_size, 64, 8, 8))

        network = (L.Conv2DLayer(network, num_filters=64, filter_size=(3, 3),
                                             nonlinearity=lasagne.nonlinearities.elu,
                                             pad="same", stride=1))
        network = (L.Conv2DLayer(network, num_filters=64, filter_size=(3, 3),
                                             nonlinearity=lasagne.nonlinearities.elu,
                                             pad="same", stride=1))

        network = L.Upscale2DLayer(network, 2, mode='repeat')
        network = (L.Conv2DLayer(network, num_filters=64, filter_size=(3, 3),
                                             nonlinearity=lasagne.nonlinearities.elu,
                                             pad="same", stride=1))
        network = (L.Conv2DLayer(network, num_filters=64, filter_size=(3, 3),
                                             nonlinearity=lasagne.nonlinearities.elu,
                                             pad="same", stride=1))

        network = L.Upscale2DLayer(network, 2, mode='repeat')
        network = (L.Conv2DLayer(network, num_filters=64, filter_size=(3, 3),
                                             nonlinearity=lasagne.nonlinearities.elu,
                                             pad="same", stride=1))
        network = (L.Conv2DLayer(network, num_filters=64, filter_size=(3, 3),
                                             nonlinearity=lasagne.nonlinearities.elu,
                                             pad="same", stride=1))

        network = L.Conv2DLayer(network, num_filters=3, filter_size=(3, 3),
                                nonlinearity=lasagne.nonlinearities.identity,
                                pad="same", stride=1)

        # output is 32 x 32
        return network

    # Trying different generator setup (more weights, bigger bottleneck, less layers)
    def build_gen6(self, input, batch_size):
        # we input the border (no noise)
        network = L.InputLayer(shape=(batch_size, 3, 64, 64), input_var=input)

        network = (L.Conv2DLayer(network, num_filters=64, filter_size=(3, 3),
                                             nonlinearity=lasagne.nonlinearities.elu,
                                             pad="same", stride=1))
        network = (L.Conv2DLayer(network, num_filters=64, filter_size=(5, 5),
                                             nonlinearity=lasagne.nonlinearities.elu,
                                             pad="same", stride=2))

        network = (L.Conv2DLayer(network, num_filters=64, filter_size=(3, 3),
                                             nonlinearity=lasagne.nonlinearities.elu,
                                             pad="same", stride=1))
        network = (L.Conv2DLayer(network, num_filters=64, filter_size=(5, 5),
                                             nonlinearity=lasagne.nonlinearities.elu,
                                             pad=1, stride=2))

        network = (L.Conv2DLayer(network, num_filters=128, filter_size=(3, 3),
                                             nonlinearity=lasagne.nonlinearities.elu,
                                             pad="same"))
        network = (L.Conv2DLayer(network, num_filters=128, filter_size=(5, 5),
                                             nonlinearity=lasagne.nonlinearities.elu,
                                             pad=1, stride=2))

        network = (L.Conv2DLayer(network, num_filters=256, filter_size=(3, 3),
                                             nonlinearity=lasagne.nonlinearities.elu,
                                             pad="same", stride=1))
        # 8x8 image size here
        network = L.DenseLayer(network, num_units=1024,
                               nonlinearity=lasagne.nonlinearities.identity)

        network = L.DenseLayer(network, num_units=8192,
                               nonlinearity=lasagne.nonlinearities.elu)

        network = L.ReshapeLayer(network, shape=(batch_size, 128, 8, 8))

        network = (L.Conv2DLayer(network, num_filters=256, filter_size=(5, 5),
                                             nonlinearity=lasagne.nonlinearities.elu,
                                             pad="same", stride=1))

        network = L.Upscale2DLayer(network, 2, mode='repeat')
        network = (L.Conv2DLayer(network, num_filters=128, filter_size=(5, 5),
                                             nonlinearity=lasagne.nonlinearities.elu,
                                             pad="same", stride=1))

        network = L.Upscale2DLayer(network, 2, mode='repeat')

        network = (L.Conv2DLayer(network, num_filters=64, filter_size=(5, 5),
                                             nonlinearity=lasagne.nonlinearities.elu,
                                             pad="same", stride=1))

        network = L.Conv2DLayer(network, num_filters=3, filter_size=(3, 3),
                                nonlinearity=lasagne.nonlinearities.identity,
                                pad="same", stride=1)
        # output is 32 x 32
        return network

    def build_dis6(self, input, batch_size):
        # we input the border (no noise)
        network = L.InputLayer(shape=(batch_size, 3, 64, 64), input_var=input)

        network = (L.Conv2DLayer(network, num_filters=48, filter_size=(5, 5),
                                             nonlinearity=lasagne.nonlinearities.elu,
                                             pad="same", stride=1))
        network = (L.Conv2DLayer(network, num_filters=48, filter_size=(3, 3),
                                             nonlinearity=lasagne.nonlinearities.elu,
                                             pad="same", stride=1))
        network = (L.Conv2DLayer(network, num_filters=48, filter_size=(3, 3),
                                             nonlinearity=lasagne.nonlinearities.elu,
                                             pad=1, stride=2))

        network = (L.Conv2DLayer(network, num_filters=64, filter_size=(3, 3),
                                             nonlinearity=lasagne.nonlinearities.elu,
                                             pad="same", stride=1))
        network = (L.Conv2DLayer(network, num_filters=64, filter_size=(3, 3),
                                             nonlinearity=lasagne.nonlinearities.elu,
                                             pad="same"))
        network = (L.Conv2DLayer(network, num_filters=64, filter_size=(3, 3),
                                             nonlinearity=lasagne.nonlinearities.elu,
                                             pad=1, stride=2))

        network = (L.Conv2DLayer(network, num_filters=128, filter_size=(3, 3),
                                             nonlinearity=lasagne.nonlinearities.elu,
                                             pad="same", stride=1))
        network = (L.Conv2DLayer(network, num_filters=128, filter_size=(3, 3),
                                             nonlinearity=lasagne.nonlinearities.elu,
                                             pad="same"))
        network = (L.Conv2DLayer(network, num_filters=128, filter_size=(3, 3),
                                             nonlinearity=lasagne.nonlinearities.elu,
                                             pad=1, stride=2))

        network = (L.Conv2DLayer(network, num_filters=192, filter_size=(3, 3),
                                             nonlinearity=lasagne.nonlinearities.elu,
                                             pad="same", stride=1))
        network = (L.Conv2DLayer(network, num_filters=192, filter_size=(3, 3),
                                             nonlinearity=lasagne.nonlinearities.elu,
                                             pad="same"))
        # 8x8 image size here
        network = L.DenseLayer(network, num_units=128,
                               nonlinearity=lasagne.nonlinearities.identity)

        network = L.DenseLayer(network, num_units=4096,
                               nonlinearity=lasagne.nonlinearities.elu)

        network = L.ReshapeLayer(network, shape=(batch_size, 64, 8, 8))

        network = (L.Conv2DLayer(network, num_filters=64, filter_size=(3, 3),
                                             nonlinearity=lasagne.nonlinearities.elu,
                                             pad="same", stride=1))
        network = (L.Conv2DLayer(network, num_filters=64, filter_size=(3, 3),
                                             nonlinearity=lasagne.nonlinearities.elu,
                                             pad="same", stride=1))

        network = L.Upscale2DLayer(network, 2, mode='repeat')
        network = (L.Conv2DLayer(network, num_filters=64, filter_size=(3, 3),
                                             nonlinearity=lasagne.nonlinearities.elu,
                                             pad="same", stride=1))
        network = (L.Conv2DLayer(network, num_filters=64, filter_size=(3, 3),
                                             nonlinearity=lasagne.nonlinearities.elu,
                                             pad="same", stride=1))

        network = L.Upscale2DLayer(network, 2, mode='repeat')
        network = (L.Conv2DLayer(network, num_filters=64, filter_size=(3, 3),
                                             nonlinearity=lasagne.nonlinearities.elu,
                                             pad="same", stride=1))
        network = (L.Conv2DLayer(network, num_filters=64, filter_size=(3, 3),
                                             nonlinearity=lasagne.nonlinearities.elu,
                                             pad="same", stride=1))

        network = L.Conv2DLayer(network, num_filters=3, filter_size=(3, 3),
                                nonlinearity=lasagne.nonlinearities.identity,
                                pad="same", stride=1)

        # output is 32 x 32
        return network

    # clone of v1 without batch norm
    def build_gen7(self, input_gen, batch_size):
        # we input the border (no noise)
        network = L.InputLayer(shape=(batch_size, 3, 64, 64), input_var=input_gen)

        network = (L.Conv2DLayer(network, num_filters=64, filter_size=(5, 5),
                                             nonlinearity=lasagne.nonlinearities.elu,
                                             pad="same", stride=1))

        network = (L.Conv2DLayer(network, num_filters=64, filter_size=(5, 5),
                                             nonlinearity=lasagne.nonlinearities.elu,
                                             pad=1, stride=2))
        network = (L.Conv2DLayer(network, num_filters=64, filter_size=(3, 3),
                                             nonlinearity=lasagne.nonlinearities.elu,
                                             pad="same", stride=1))
        network = (L.Conv2DLayer(network, num_filters=64, filter_size=(3, 3),
                                             nonlinearity=lasagne.nonlinearities.elu,
                                             pad="same"))

        network = (L.Conv2DLayer(network, num_filters=128, filter_size=(3, 3),
                                             nonlinearity=lasagne.nonlinearities.elu,
                                             pad=1, stride=2))
        network = (L.Conv2DLayer(network, num_filters=128, filter_size=(3, 3),
                                             nonlinearity=lasagne.nonlinearities.elu,
                                             pad="same", stride=1))
        network = (L.Conv2DLayer(network, num_filters=128, filter_size=(3, 3),
                                             nonlinearity=lasagne.nonlinearities.elu,
                                             pad="same"))

        network = (L.Conv2DLayer(network, num_filters=128, filter_size=(3, 3),
                                             nonlinearity=lasagne.nonlinearities.elu,
                                             pad=1, stride=2))
        network = (L.Conv2DLayer(network, num_filters=128, filter_size=(3, 3),
                                             nonlinearity=lasagne.nonlinearities.elu,
                                             pad="same", stride=1))
        network = (L.Conv2DLayer(network, num_filters=128, filter_size=(3, 3),
                                             nonlinearity=lasagne.nonlinearities.elu,
                                             pad="same"))
        # 8x8 image size here
        network = L.DenseLayer(network, num_units=128,
                               nonlinearity=lasagne.nonlinearities.identity)

        network = L.DenseLayer(network, num_units=4096,
                               nonlinearity=lasagne.nonlinearities.elu)

        network = L.ReshapeLayer(network, shape=(batch_size, 64, 8, 8))

        network = (L.Conv2DLayer(network, num_filters=64, filter_size=(3, 3),
                                             nonlinearity=lasagne.nonlinearities.elu,
                                             pad="same", stride=1))
        network = (L.Conv2DLayer(network, num_filters=64, filter_size=(3, 3),
                                             nonlinearity=lasagne.nonlinearities.elu,
                                             pad="same", stride=1))

        network = L.Upscale2DLayer(network, 2, mode='repeat')
        network = (L.Conv2DLayer(network, num_filters=64, filter_size=(3, 3),
                                             nonlinearity=lasagne.nonlinearities.elu,
                                             pad="same", stride=1))
        network = (L.Conv2DLayer(network, num_filters=64, filter_size=(3, 3),
                                             nonlinearity=lasagne.nonlinearities.elu,
                                             pad="same", stride=1))

        network = L.Upscale2DLayer(network, 2, mode='repeat')
        network = (L.Conv2DLayer(network, num_filters=64, filter_size=(3, 3),
                                             nonlinearity=lasagne.nonlinearities.elu,
                                             pad="same", stride=1))
        network = (L.Conv2DLayer(network, num_filters=64, filter_size=(3, 3),
                                             nonlinearity=lasagne.nonlinearities.elu,
                                             pad="same", stride=1))


        network = L.Conv2DLayer(network, num_filters=3, filter_size=(3, 3),
                                nonlinearity=lasagne.nonlinearities.leaky_rectify,
                                pad="same", stride=1)

        # output is 32 x 32
        return network

    def build_dis7(self, input_dis, batch_size):
        # We input only the 32x32 center patch
        network = L.InputLayer(shape=(batch_size, 3, 32, 32), input_var=input_dis)

        network = (L.Conv2DLayer(network, num_filters=64, filter_size=(3, 3),
                                             nonlinearity=lasagne.nonlinearities.elu,
                                             pad="same", stride=1))
        network = (L.Conv2DLayer(network, num_filters=64, filter_size=(3, 3),
                                             nonlinearity=lasagne.nonlinearities.elu,
                                             pad="same", stride=1))

        network = (L.Conv2DLayer(network, num_filters=128, filter_size=(3, 3),
                                             nonlinearity=lasagne.nonlinearities.elu,
                                             pad=1, stride=2))
        network = (L.Conv2DLayer(network, num_filters=128, filter_size=(3, 3),
                                             nonlinearity=lasagne.nonlinearities.elu,
                                             pad="same", stride=1))
        network = (L.Conv2DLayer(network, num_filters=128, filter_size=(3, 3),
                                             nonlinearity=lasagne.nonlinearities.elu,
                                             pad="same"))

        network = (L.Conv2DLayer(network, num_filters=128, filter_size=(3, 3),
                                             nonlinearity=lasagne.nonlinearities.elu,
                                             pad=1, stride=2))
        network = (L.Conv2DLayer(network, num_filters=128, filter_size=(3, 3),
                                             nonlinearity=lasagne.nonlinearities.elu,
                                             pad="same", stride=1))
        network = (L.Conv2DLayer(network, num_filters=128, filter_size=(3, 3),
                                             nonlinearity=lasagne.nonlinearities.elu,
                                             pad="same"))

        # 8x8 image size here
        network = L.DenseLayer(network, num_units=128,
                               nonlinearity=lasagne.nonlinearities.identity)

        network = L.DenseLayer(network, num_units=4096,
                               nonlinearity=lasagne.nonlinearities.elu)

        network = L.ReshapeLayer(network, shape=(batch_size, 64, 8, 8))

        network = (L.Conv2DLayer(network, num_filters=64, filter_size=(3, 3),
                                             nonlinearity=lasagne.nonlinearities.elu,
                                             pad="same", stride=1))
        network = (L.Conv2DLayer(network, num_filters=64, filter_size=(3, 3),
                                             nonlinearity=lasagne.nonlinearities.elu,
                                             pad="same", stride=1))

        network = L.Upscale2DLayer(network, 2, mode='repeat')
        network = (L.Conv2DLayer(network, num_filters=64, filter_size=(3, 3),
                                             nonlinearity=lasagne.nonlinearities.elu,
                                             pad="same", stride=1))
        network = (L.Conv2DLayer(network, num_filters=64, filter_size=(3, 3),
                                             nonlinearity=lasagne.nonlinearities.elu,
                                             pad="same", stride=1))

        network = L.Upscale2DLayer(network, 2, mode='repeat')
        network = (L.Conv2DLayer(network, num_filters=64, filter_size=(3, 3),
                                             nonlinearity=lasagne.nonlinearities.elu,
                                             pad="same", stride=1))
        network = (L.Conv2DLayer(network, num_filters=64, filter_size=(3, 3),
                                             nonlinearity=lasagne.nonlinearities.elu,
                                             pad="same", stride=1))

        network = L.Conv2DLayer(network, num_filters=3, filter_size=(3, 3),
                                nonlinearity=lasagne.nonlinearities.leaky_rectify,
                                pad="same", stride=1)

        return network

    # use captions now
    def build_gen8(self, input_gen, input_caption, batch_size):
        # we input the border (no noise)
        network = L.InputLayer(shape=(batch_size, 3, 64, 64), input_var=input_gen)

        network = (L.Conv2DLayer(network, num_filters=64, filter_size=(5, 5),
                                             nonlinearity=lasagne.nonlinearities.elu,
                                             pad="same", stride=1))

        network = (L.Conv2DLayer(network, num_filters=64, filter_size=(5, 5),
                                             nonlinearity=lasagne.nonlinearities.elu,
                                             pad=1, stride=2))
        network = (L.Conv2DLayer(network, num_filters=64, filter_size=(3, 3),
                                             nonlinearity=lasagne.nonlinearities.elu,
                                             pad="same", stride=1))
        network = (L.Conv2DLayer(network, num_filters=64, filter_size=(3, 3),
                                             nonlinearity=lasagne.nonlinearities.elu,
                                             pad="same"))

        network = (L.Conv2DLayer(network, num_filters=96, filter_size=(3, 3),
                                             nonlinearity=lasagne.nonlinearities.elu,
                                             pad=1, stride=2))
        network = (L.Conv2DLayer(network, num_filters=96, filter_size=(3, 3),
                                             nonlinearity=lasagne.nonlinearities.elu,
                                             pad="same", stride=1))
        network = (L.Conv2DLayer(network, num_filters=96, filter_size=(3, 3),
                                             nonlinearity=lasagne.nonlinearities.elu,
                                             pad="same"))

        network = (L.Conv2DLayer(network, num_filters=128, filter_size=(3, 3),
                                             nonlinearity=lasagne.nonlinearities.elu,
                                             pad=1, stride=2))
        network = (L.Conv2DLayer(network, num_filters=128, filter_size=(3, 3),
                                             nonlinearity=lasagne.nonlinearities.elu,
                                             pad="same", stride=1))
        network = (L.Conv2DLayer(network, num_filters=128, filter_size=(3, 3),
                                             nonlinearity=lasagne.nonlinearities.elu,
                                             pad="same"))

        # 8x8 image size here
        network = L.DenseLayer(network, num_units=128,
                               nonlinearity=lasagne.nonlinearities.identity)

        # Reduce dimensionality of embedding and concatenate with hidden rep
        network2 = L.InputLayer(shape=(batch_size, 4800), input_var=input_caption)
        network2 = L.DenseLayer(network2, num_units=128,
                               nonlinearity=lasagne.nonlinearities.identity)
        network = L.ConcatLayer([network, network2])

        network = L.DenseLayer(network, num_units=4096,
                               nonlinearity=lasagne.nonlinearities.elu)

        network = L.ReshapeLayer(network, shape=(batch_size, 64, 8, 8))

        network = (L.Conv2DLayer(network, num_filters=64, filter_size=(3, 3),
                                             nonlinearity=lasagne.nonlinearities.elu,
                                             pad="same", stride=1))
        network = (L.Conv2DLayer(network, num_filters=64, filter_size=(3, 3),
                                             nonlinearity=lasagne.nonlinearities.elu,
                                             pad="same", stride=1))

        network = L.Upscale2DLayer(network, 2, mode='repeat')
        network = (L.Conv2DLayer(network, num_filters=64, filter_size=(3, 3),
                                             nonlinearity=lasagne.nonlinearities.elu,
                                             pad="same", stride=1))
        network = (L.Conv2DLayer(network, num_filters=64, filter_size=(3, 3),
                                             nonlinearity=lasagne.nonlinearities.elu,
                                             pad="same", stride=1))

        network = L.Upscale2DLayer(network, 2, mode='repeat')
        network = (L.Conv2DLayer(network, num_filters=64, filter_size=(3, 3),
                                             nonlinearity=lasagne.nonlinearities.elu,
                                             pad="same", stride=1))
        network = (L.Conv2DLayer(network, num_filters=64, filter_size=(3, 3),
                                             nonlinearity=lasagne.nonlinearities.elu,
                                             pad="same", stride=1))

        network = L.Conv2DLayer(network, num_filters=3, filter_size=(3, 3),
                                nonlinearity=lasagne.nonlinearities.identity,
                                pad="same", stride=1)

        # output is 32 x 32
        return network

    def build_dis8(self, input_dis, input_caption, batch_size):
        # We input only the 32x32 center patch
        network = L.InputLayer(shape=(batch_size, 3, 32, 32), input_var=input_dis)

        network = (L.Conv2DLayer(network, num_filters=64, filter_size=(3, 3),
                                             nonlinearity=lasagne.nonlinearities.elu,
                                             pad="same", stride=1))
        network = (L.Conv2DLayer(network, num_filters=64, filter_size=(3, 3),
                                             nonlinearity=lasagne.nonlinearities.elu,
                                             pad="same", stride=1))
        network = (L.Conv2DLayer(network, num_filters=64, filter_size=(3, 3),
                                             nonlinearity=lasagne.nonlinearities.elu,
                                             pad=1, stride=2))

        network = (L.Conv2DLayer(network, num_filters=96, filter_size=(3, 3),
                                             nonlinearity=lasagne.nonlinearities.elu,
                                             pad="same", stride=1))
        network = (L.Conv2DLayer(network, num_filters=96, filter_size=(3, 3),
                                             nonlinearity=lasagne.nonlinearities.elu,
                                             pad="same"))
        network = (L.Conv2DLayer(network, num_filters=96, filter_size=(3, 3),
                                             nonlinearity=lasagne.nonlinearities.elu,
                                             pad=1, stride=2))

        network = (L.Conv2DLayer(network, num_filters=128, filter_size=(3, 3),
                                             nonlinearity=lasagne.nonlinearities.elu,
                                             pad="same", stride=1))
        network = (L.Conv2DLayer(network, num_filters=128, filter_size=(3, 3),
                                             nonlinearity=lasagne.nonlinearities.elu,
                                             pad="same"))

        # 8x8 image size here
        network = L.DenseLayer(network, num_units=128,
                               nonlinearity=lasagne.nonlinearities.identity)

        # Reduce dimensionality of embedding and concatenate with hidden rep
        network2 = L.InputLayer(shape=(batch_size, 4800), input_var=input_caption)
        network2 = L.DenseLayer(network2, num_units=128,
                               nonlinearity=lasagne.nonlinearities.identity)
        network = L.ConcatLayer([network, network2])

        network = L.DenseLayer(network, num_units=4096,
                               nonlinearity=lasagne.nonlinearities.elu)

        network = L.ReshapeLayer(network, shape=(batch_size, 64, 8, 8))

        network = (L.Conv2DLayer(network, num_filters=64, filter_size=(3, 3),
                                             nonlinearity=lasagne.nonlinearities.elu,
                                             pad="same", stride=1))
        network = (L.Conv2DLayer(network, num_filters=64, filter_size=(3, 3),
                                             nonlinearity=lasagne.nonlinearities.elu,
                                             pad="same", stride=1))

        network = L.Upscale2DLayer(network, 2, mode='repeat')
        network = (L.Conv2DLayer(network, num_filters=64, filter_size=(3, 3),
                                             nonlinearity=lasagne.nonlinearities.elu,
                                             pad="same", stride=1))
        network = (L.Conv2DLayer(network, num_filters=64, filter_size=(3, 3),
                                             nonlinearity=lasagne.nonlinearities.elu,
                                             pad="same", stride=1))

        network = L.Upscale2DLayer(network, 2, mode='repeat')
        network = (L.Conv2DLayer(network, num_filters=64, filter_size=(3, 3),
                                             nonlinearity=lasagne.nonlinearities.elu,
                                             pad="same", stride=1))
        network = (L.Conv2DLayer(network, num_filters=64, filter_size=(3, 3),
                                             nonlinearity=lasagne.nonlinearities.elu,
                                             pad="same", stride=1))

        network = L.Conv2DLayer(network, num_filters=3, filter_size=(3, 3),
                                nonlinearity=lasagne.nonlinearities.identity,
                                pad="same", stride=1)

        return network


