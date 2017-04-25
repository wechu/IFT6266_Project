import theano
from theano import tensor as T
import numpy as np
import numpy.random as rng
import lasagne
from lasagne import layers as L
import pickle, gzip
import matplotlib.pyplot as plt
import time
#test
# Custom files
import utils
import ConvNetBase
import GAN
import ContextEncoder

theano.config.exception_verbosity = "high"
theano.config.allow_gc = True

if __name__ == "__main__":
    log_file = "conlog1.txt"

    with open(log_file, 'w') as f:
        f.write("Start!\n")
        f.write(str(theano.config.device) + "\n")

    print("Get data...")
    raw_x_train, raw_y_train, raw_cap_train = utils.load_dataset("Data/train2014.pkl.gz")
    raw_x_valid, raw_y_valid, raw_cap_valid = utils.load_dataset("Data/val2014.pkl.gz")

    print(raw_x_train.shape)
    print(raw_x_valid.shape)

    x_train_num = raw_x_train.shape[0]
    x_valid_num = raw_x_valid.shape[0]

    with open(log_file, 'a') as f:
        f.write("Got data\n")

    # Fix shared variables later
    x_train = raw_x_train
    y_train = raw_y_train
    x_valid = raw_x_valid
    y_valid = raw_y_valid

    # Load from iter
    name = "CON9"
    net = ContextEncoder.BEGAN(name, gamma=0.3, k_initial=0.0, loss_adv=0.9, dis_type="center_to_center")
    net.compile_theano(0.000004, batch_size=64, mode="v7", momentum=0.5, training=True,
                       inputs_shape=x_train.shape[1:], targets_shape=y_train.shape[1:], macro_batch_size=2048)

    it = 92000
    if it > 0:
        # utils.load_model_weights(net.gen, "{}_gen_weights_it_{}.pkl".format(name, it))
        # utils.load_model_weights(net.dis, "{}_dis_weights_it_{}.pkl".format(name, it))
        utils.load_updates(net.updates, "{}_updates_it_{}.pkl".format(name, it))

    with open(log_file, 'a') as f:
        f.write("done compiling\n")
        f.write("Start training\n")

    net.train(x_train, y_train, batch_size=64, macro_batch_size=2048, anneal=True, start_iter=(it+1), log_file=log_file)

    #net.train(y_train, noise_type="uniform", log_file=log_file)  # train GAN on 32x32 center patches
    #net.train(y_train, noise_type="gaussian", start_iter=26401, log_file=log_file)

    # training loop
    # num_epochs=50
    # iter = 1
    # index_list = np.arange(x_train_num, dtype='int32') #used for shuffling
    # train_file = net.name + "_train.txt"
    # valid_file = net.name + "_valid.txt"
    #
    # print("Start training...")
    # time_start = time.time()
    # for epoch in range(num_epochs):
    #     with open(log_file, 'a') as f:
    #         f.write("----- Epoch" + str(epoch) + " " + str(time.time() - time_start) + " -----\n")
    #
    #     # Shuffle examples every epoch
    #     np.random.shuffle(index_list)
    #
    #     for i in range(int(x_train_num / batch_size)):
    #         shuffle_ind = index_list[i*batch_size:(i+1)*batch_size]
    #         train_loss, train_pred = net.train_method(x_train[shuffle_ind], y_train[shuffle_ind])
    #
    #         with open(log_file, 'a') as f:
    #             f.write(str(epoch) + " " + str(i) + "\t" + str(float(train_loss)) + "\n")
    #         with open(train_file, 'a') as f:
    #             f.write(str(float(train_loss)) + "\n")
    #
    #         # Validation set
    #         if iter % 600 == 0:
    #             valid_losses = []
    #             for j in range(int(x_valid_num / batch_size)):
    #                 res2 = net.valid_method(x_valid[j*batch_size:(j+1)*batch_size], y_valid[j*batch_size:(j+1)*batch_size])
    #                 valid_losses.append(res2[0])
    #
    #             with open(log_file, 'a') as f:
    #                 f.write("\t\tValid:\t" + str(float(np.mean(valid_losses))) + "\n")
    #             with open(valid_file, 'a') as f:
    #                 f.write(str(float(np.mean(valid_losses))) + "\n")
    #
    #             utils.save_model_weights(net.model, "{}_weights_e_{}_it_{}.pkl".format(net.name, epoch, iter))
    #
    #         iter += 1

    # print("Total time", time.time() - time_start)
    # test a few values

    with open(log_file, 'a') as f:
        #f.write("Total time" + str(time.time() - time_start) + "\n")
        f.write("Done!\n")
