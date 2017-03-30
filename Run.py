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

theano.config.exception_verbosity = "high"
theano.config.allow_gc = False


if __name__ == "__main__":
    log_file = "log.txt"

    with open(log_file, 'w') as f:
        f.write("Start!\n")
        f.write(str(theano.config.device) + "\n")

    print("Get data...")
    raw_x_train, raw_y_train, raw_cap_train = utils.load_dataset("Data/train2014.pkl.gz")
    raw_x_valid, raw_y_valid, raw_cap_valid = utils.load_dataset("Data/val2014.pkl.gz")

    raw_x_train = np.array(raw_x_train, dtype='float32') /255 # rescale data to [0,1]
    raw_y_train = np.array(raw_y_train, dtype='float32') /255
    raw_x_valid = np.array(raw_x_valid, dtype='float32') /255
    raw_y_valid = np.array(raw_y_valid, dtype='float32') /255

    # Rearrange the dims in the data
    raw_x_train = raw_x_train.transpose((0, 3, 1, 2))  # put the channel dim before the horiz/verti dims
    raw_y_train = raw_y_train.transpose((0, 3, 1, 2))
    raw_x_valid = raw_x_valid.transpose((0, 3, 1, 2))
    raw_y_valid = raw_y_valid.transpose((0, 3, 1, 2))

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

    input_var = T.tensor4('inputs', dtype='float32')
    target_var = T.tensor4('targets', dtype='float32')

    # build network
    network = ConvNetBase.build_cnn(input_var, batch_size=64)

    with open(log_file, 'a') as f:
        f.write("built network\n")

    # get prediction and loss
    pred = L.get_output(network)
    loss = lasagne.objectives.squared_error(pred, target_var).mean()

    # with open(log_file, 'a') as f:
    #     theano.printing.debugprint(pred, file=f)

    # get test predictions and loss
    test_pred = L.get_output(network, deterministic=True)
    test_loss = lasagne.objectives.squared_error(test_pred, target_var)
    test_loss = test_loss.mean()

    # get updates
    lr = theano.shared(np.array(0.001, dtype='float32'))

    params = L.get_all_params(network, trainable=True)
    updates = lasagne.updates.adam(loss, params, learning_rate=lr)

    with open(log_file, 'a') as f:
        f.write("Compiling methods... ")

    print("Compile methods...")
    # compile methods
    train_method = theano.function([input_var, target_var], [loss, pred], updates=updates)
    valid_method = theano.function([input_var, target_var], [test_loss, test_pred])

    with open(log_file, 'a') as f:
        f.write("done compiling\n")
        f.write("Start training\n")

    # training loop
    batch_size=64
    num_epochs= 3
    # iter = 1
    # cooldown = 50
    # train_losses = []
    # moving_avg = [0] * 10
    print("Start training...")

    time_start = time.time()
    for epoch in range(num_epochs):
        with open(log_file, 'a') as f:
            f.write("----- Epoch" + str(epoch) + " " + str(time.time() - time_start) + " -----\n")

        for i in range(int(x_train_num / batch_size)):
            train_loss, train_pred = train_method(x_train[i*batch_size:(i+1)*batch_size], y_train[i*batch_size:(i+1)*batch_size])

            with open(log_file, 'a') as f:
                f.write(str(epoch) + " " + str(i) + "\t" + str(float(train_loss)) + "\n")

            # if cooldown < 0:
            #     if moving_avg[-1] / moving_avg[-2] > 0.9999: # if there's no progress, reduce learning rate
            #         lr.set_value(np.array(0.5, dtype='float32') * lr.get_value())
            #         print("Reduced learning rate", lr.get_value())
            #         cooldown = 300
            #
            # train_losses.append(float(train_loss))
            # moving_avg.append(sum(train_losses[-10:]))
            # del train_losses[0]
            #
            # iter += 1
            # cooldown -= 1

            # Validation set
            if (i+1) % 450 == 0 or i == int(x_train_num / batch_size):
                valid_losses = []
                for j in range(int(x_valid_num / batch_size)):
                    res2 = valid_method(x_valid[j*batch_size:(j+1)*batch_size], y_valid[j*batch_size:(j+1)*batch_size])
                    valid_losses.append(res2[0])

                with open(log_file, 'a') as f:
                    f.write("\t\tValid:\t" + str(float(np.mean(valid_losses))) + "\n")

                utils.save_model_weights(network, "weights_e_{}_i_{}.pkl".format(epoch, i))


    print("Total time", time.time() - time_start)
    # test a few values

    with open(log_file, 'a') as f:
        f.write("Total time" + str(time.time() - time_start) + "\n")
        f.write("Done!\n")
