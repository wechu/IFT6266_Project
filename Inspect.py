import theano
from theano import tensor as T
import numpy as np
import numpy.random as rng
import lasagne
from lasagne import layers as L
import pickle, gzip
import matplotlib.pyplot as plt
import time

# Custom files
import utils
import ConvNetBase
import GAN
import ContextEncoder


def squeeze_range(x):
    x = 0 if x < 0 else x
    x = 1 if x > 1 else x
    return x
squeeze = np.vectorize(squeeze_range)


def draw_image(input2, target2, pred2):
    """ Draws the true image and the predicted image """
    input = input2[:, :, :] # make deep copy
    target = target2[:, :, :]
    pred = pred2[:, :, :]
    # dimshuffle to put channels in last dim so we can plot (x, y, channel)
    input = input.transpose((1, 2, 0))
    target = target.transpose((1, 2, 0))
    pred = squeeze(pred)
    pred = pred.transpose((1, 2, 0))

    #print(input.shape)
    # Place the center back of the true image
    center = (int(np.floor(input.shape[0] / 2.)), int(np.floor(input.shape[1] / 2.)))

    #print(center)
    input[center[0] - 16:center[0] + 16, center[1] - 16:center[1] + 16, :] = target

    # Draw the target
    plt.figure(figsize=(6,6)).canvas.set_window_title("Target")
    plt.axis('off')
    plt.imshow(input)

    # Place the center back of the predicted image
    # Draw the prediction
    input[center[0] - 16:center[0] + 16, center[1] - 16:center[1] + 16, :] = pred
    plt.figure(figsize=(6,6)).canvas.set_window_title("Prediction")
    plt.axis('off')
    plt.imshow(input)

    plt.show()
    return

def draw_gan_image(image):
    image = squeeze(image)
    image = image.transpose((1, 2, 0))

    plt.figure(figsize=(8, 8)).canvas.set_window_title("Generated")
    plt.axis('off')
    plt.imshow(image)
    plt.show()
    return

def plot_training_curves(train_loss, valid_loss=None, valid_freq=0, skip=0):
    """ Plots the losses. Use plt.show() after this method. """
    plt.plot(range(skip, len(train_loss)), train_loss[skip:])
    if valid_loss is not None:
        plt.plot(valid_freq*np.array(range(1,1+len(valid_loss))), valid_loss)

    return

if __name__ == '__main__':
    mode = "plot_gan"

    if mode == "plot_loss":
        # plot losses
        with open("Results/CNN4/CNN4_train.txt") as f:
            train_losses = np.array(f.readlines(), dtype='float64')
        with open("Results/CNN4/CNN4_valid.txt") as f:
            valid_losses = np.array(f.readlines(), dtype='float64')

        valid_freq = 600
        print(valid_losses[0:5])
        print("Min valid:", np.min(valid_losses), np.argmin(valid_losses) * valid_freq,
              "\tTrain:", train_losses[np.argmin(valid_losses) * valid_freq])
        plot_training_curves(train_losses, valid_losses, valid_freq, skip=100)
        plt.show()

    if mode == "plot_gan":
        # plot losses
        model_name = "CON9"
        with open("Results/" + model_name + "_dis.txt") as f:
            dis_losses = np.array(f.readlines(), dtype='float64')
        with open("Results/" + model_name + "_gen.txt") as f:
            gen_losses = np.array(f.readlines(), dtype='float64')
        with open("Results/" + model_name + "_check.txt") as f:
            check_losses = np.array(f.readlines(), dtype='float64')

        skip = 300
        #plot_training_curves(check_losses, skip=skip)  # blue
        #plot_training_curves(dis_losses, skip=skip)  # green
        #plot_training_curves(gen_losses, skip=skip)  # red

        window = 150
        avg_check = [np.mean(check_losses[max(i-window, 0):i+1]) for i in range(len(check_losses))]
        plot_training_curves(avg_check[:100000], skip=skip)
        #plt.ylim((0.05, 0.07))

        avg_gen = [np.mean(gen_losses[max(i-window, 0):i+1]) for i in range(len(gen_losses))]
        plot_training_curves(avg_gen[:100000], skip=skip)

        avg_dis = [np.mean(dis_losses[max(i-window, 0):i+1]) for i in range(len(dis_losses))]
        plot_training_curves(avg_dis[:100000], skip=skip)

        plt.show()


    if mode == "try_gan":
        net = GAN.BEGAN("", noise_dim=64)

        net.compile_theano(learn_rate=0, batch_size=64, mode="v1")
        utils.load_model_weights(net.gen, "Results/BEG_gen_weights_e_17_it_22800.pkl")
        noise = np.random.normal(size=(64, 64)).astype('float32')
        print("Generating...")
        samples = net.gen_method(noise)

        for i in range(10):
            draw_gan_image(samples[i])


    if mode == "try_con":
        # Get data
        print("Get data...")
        check_train = False
        if check_train:
            x_train, y_train, cap_train = utils.load_dataset("Data/train2014.pkl.gz")
        x_valid, y_valid, cap_valid = utils.load_dataset("Data/val2014.pkl.gz")

        net = ContextEncoder.BEGAN("", gamma=0.5, k_initial=0, use_caption=False)
        net.compile_theano(mode="v7", batch_size=32, learn_rate=0, training=False)
        #utils.load_model_weights(net.gen, "Results/CON8_3_gen_weights_it_70000.pkl")
        utils.load_model_weights(net.gen, "Results/CON9_gen_weights_it_100000.pkl")

        # Test examples
        batch_size = 32
        ex = 290  # batch number to check 530, 640

        ex_x_valid = np.copy(x_valid[ ex *batch_size:( ex +1 ) *batch_size])
        ex_y_valid = np.copy(y_valid[ ex *batch_size:( ex +1 ) *batch_size])

        predv = net.gen_method(ex_x_valid)  # from valid set
        for i in range(0, 5):
            draw_image(ex_x_valid[i], ex_y_valid[i], predv[i])

        if check_train:
            i=1
            ex_x_train = x_train[ ex *batch_size:( ex +1 ) *batch_size]
            ex_y_train = y_train[ ex *batch_size:( ex +1 ) *batch_size]
            pred = net.gen_method(ex_x_train)  # from train set
            draw_image(ex_x_train[i], ex_y_train[i], pred[i])

    if mode == "try_cop":
        # Get data
        print("Get data...")
        check_train = False
        if check_train:
            x_train, y_train, cap_train = utils.load_dataset("Data/train2014.pkl.gz")
        x_valid, y_valid, cap_valid = utils.load_dataset("Data/val2014.pkl.gz")

        with open("Data/val2014_embed_all.pkl", 'rb') as f:
            embed_valid = pickle.load(f, encoding='latin1')

        net = ContextEncoder.BEGAN("", gamma=0.5, k_initial=0, use_caption=True)
        net.compile_theano(mode="v8", batch_size=32, learn_rate=0, training=False)
        # utils.load_model_weights(net.gen, "Results/CON8_3_gen_weights_it_70000.pkl")
        utils.load_model_weights(net.gen, "Results/COP1_gen_weights_it_106000.pkl")

        # Test examples
        batch_size = 32
        ex = 290  # batch number to check

        ex_x_valid = np.copy(x_valid[ex * batch_size:(ex + 1) * batch_size])
        ex_y_valid = np.copy(y_valid[ex * batch_size:(ex + 1) * batch_size])
        ex_embed_valid = np.copy(np.array(embed_valid[ex * batch_size:(ex+1)*batch_size]))
        ex_cap_valid = cap_valid[ex * batch_size:(ex+1)*batch_size]

        cap_index = 0
        predv = net.gen_method(ex_x_valid, ex_embed_valid[:, cap_index])  # from valid set

        for i in range(0, 5):
            print(ex_cap_valid[i][cap_index])
            draw_image(ex_x_valid[i], ex_y_valid[i], predv[i])

        if check_train:
            i=1
            ex_x_train = x_train[ ex *batch_size:( ex +1 ) *batch_size]
            ex_y_train = y_train[ ex *batch_size:( ex +1 ) *batch_size]
            pred = net.gen_method(ex_x_train)  # from train set
            draw_image(ex_x_train[i], ex_y_train[i], pred[i])
