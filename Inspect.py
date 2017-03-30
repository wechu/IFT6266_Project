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




def draw_image(input, target, pred):
    """ Draws the true image and the predicted image """
    # dimshuffle to put channels in last dim so we can plot (x, y, channel)
    input = input.transpose((1, 2, 0))
    target = target.transpose((1, 2, 0))
    pred = pred.transpose((1, 2, 0))

    print(input.shape)
    # Place the center back of the true image
    center = (int(np.floor(input.shape[0] / 2.)), int(np.floor(input.shape[1] / 2.)))

    print(center)
    input[center[0] - 16:center[0] + 16, center[1] - 16:center[1] + 16, :] = target

    # Draw the target
    plt.figure(figsize=(8,8)).canvas.set_window_title("Target")
    plt.imshow(input)

    # Place the center back of the predicted image
    # Draw the prediction
    input[center[0] - 16:center[0] + 16, center[1] - 16:center[1] + 16, :] = pred
    plt.figure(figsize=(8,8)).canvas.set_window_title("Prediction")
    plt.imshow(input)

    plt.show()
    return

def plot_training_curves(train_loss, valid_loss):
    """ Plots the training and validation set losses """

    return

def squeeze_range(x):
    x = 0 if x < 0 else x
    x = 1 if x > 1 else x
    return x
squeeze = np.vectorize(squeeze_range)


#TODO load model
#TODO fix this



# t_x = np.random.uniform(0, 1, (3, 64, 64)).astype('float32')
# t_y = np.random.uniform(0, 1, (3, 32, 32)).astype('float32')
# t_p = np.zeros((3, 32, 32), dtype='float32')
# draw_image(t_x, t_y, t_p)
# quit()

if __name__ == '__main__':
    # Get data
    print("Get data...")
    x_train, y_train, cap_train = utils.load_dataset("Data/train2014.pkl.gz")
    x_valid, y_valid, cap_valid = utils.load_dataset("Data/val2014.pkl.gz")

    x_train = np.array(x_train, dtype='float32') /255  # rescale data to [0,1]
    y_train = np.array(y_train, dtype='float32') /255
    x_valid = np.array(x_valid, dtype='float32') /255
    y_valid = np.array(y_valid, dtype='float32') /255

    # Rearrange the dims in the data
    x_train = x_train.transpose((0, 3, 1, 2))  # put the channel dim before the horiz/verti dims
    y_train = y_train.transpose((0, 3, 1, 2))
    x_valid = x_valid.transpose((0, 3, 1, 2))
    y_valid = y_valid.transpose((0, 3, 1, 2))

    # initialize model
    input_var = T.tensor4('inputs', dtype='float32')
    target_var = T.tensor4('targets', dtype='float32')
    network = ConvNetBase.build_cnn(input_var, batch_size=64)

    # get test predictions and loss
    test_pred = L.get_output(network, deterministic=True)
    test_loss = lasagne.objectives.squared_error(test_pred, target_var)
    test_loss = test_loss.mean()

    # compile method
    valid_method = theano.function([input_var, target_var], [test_loss, test_pred])

    # Load model from file
    utils.load_model_weights(network, "Results/weights_e_2_i_899.pkl")

    # Test examples
    batch_size = 64
    ex = 0  # batch number to check

    ex_x_train = x_train[ ex *batch_size:( ex +1 ) *batch_size]
    ex_y_train = y_train[ ex *batch_size:( ex +1 ) *batch_size]
    ex_x_valid = x_valid[ ex *batch_size:( ex +1 ) *batch_size]
    ex_y_valid = y_valid[ ex *batch_size:( ex +1 ) *batch_size]

    _, pred = valid_method(ex_x_train, ex_y_train)  # from train set
    _, predv = valid_method(ex_x_valid, ex_y_valid)  # from valid set

    np.set_printoptions(threshold=np.nan)

    pred = squeeze(pred)
    predv = squeeze(predv)

    i=2
    draw_image(ex_x_train[i], ex_y_train[i], pred[i])
    draw_image(ex_x_valid[i], ex_y_valid[i], predv[i])


    # t_x = np.random.uniform(0, 1, (3, 32, 32)).astype('float32')
    # t_y = np.random.uniform(0, 1, (3, 32, 32)).astype('float32')
    # t_p = np.zeros((3, 32, 32), dtype='float32')


    # #####################################################
    # pred = pred.transpose((0, 2, 3, 1))  # dimshuffle to put channels in last dim so we can plot (batch, x, y, channel)
    # pred2 = pred2.transpose((0, 2, 3, 1))
    #
    # # process predictions to be able to plot
    # pred = pred.astype('int32')
    # pred2 = pred2.astype('int32')
    # def squeeze_range(x):
    #     x = 0 if x < 0 else x
    #     x = 255 if x > 255 else x
    #     return x
    # squeeze = np.vectorize(squeeze_range)
    # pred = squeeze(pred)
    # pred2 = squeeze(pred2)
    # # convert to uint8 for plotting (don't need this if pixels are in [0, 1])
    # pred = pred.astype('uint8')
    # pred2 = pred2.astype('uint8')
    #
    # pred = pred.transpose((0, 2, 3, 1))  # dimshuffle to put channels in last dim so we can plot (batch, x, y, channel)
    # pred2 = pred2.transpose((0, 2, 3, 1))
    # ex_x_train = ex_x_train.transpose((0, 2, 3, 1))
    # ex_y_train = ex_y_train.transpose((0, 2, 3, 1))
    # ex_x_valid = ex_x_valid.transpose((0, 2, 3, 1))
    # ex_y_valid = ex_y_valid.transpose((0, 2, 3, 1))
    #
    #
    # plt.rcParams['toolbar'] = 'None'
    #
    #
    #
    #
    #
    #
    #
    #
    #
    # for i in range(min(batch_size, 6)):
    #     residuals = pred[i].astype('int32') - ex_y_train[i].astype('int32')
    #
    #     # plt.figure()
    #     # print(pred[i].flatten().shape)
    #     plt.hist(pred[i].flatten())
    #     # plt.figure()
    #     # plt.hist(residuals.flatten())
    #     print(np.mean(np.square(residuals.flatten())))
    #     # print(pred[i].dtype)
    #     # print(ex_y_train[i].dtype)
    #     # print("residuals")
    #     # print(residuals)
    #     # print("prediction")
    #     # print(pred[i])
    #     # print("true y")
    #     # print(ex_y_train[i])
    #
    #
    #
    #
    #     plt.figure(figsize=(8 ,8))
    #     plt.imshow(ex_x_train[i])
    #     plt.figure(figsize=(4 ,4)).canvas.set_window_title("Target")
    #     plt.imshow(ex_y_train[i])
    #
    #     ran_res = np.max(residuals) - np.min(residuals)
    #     if ran_res != 0:
    #         plt.figure(figsize=(4 ,4))
    #         plt.imshow((residuals - np.min(residuals) ) /ran_res)
    #     else:
    #         print("All residuals are 0")
    #
    #     plt.figure(figsize=(4 ,4)).canvas.set_window_title("Prediction")
    #     plt.imshow(pred[i])
    #
    #
    #
    #     plt.show()
    #
    # print("Validation test")
    # for i in range(batch_size):
    #     residuals = pred2[i].astype('int32') - ex_y_valid[i].astype('int32')
    #
    #     # plt.figure()
    #     # print(pred2[i].flatten().shape)
    #     # plt.hist(pred2[i].flatten())
    #     plt.figure()
    #     plt.hist(residuals.flatten())
    #     print(np.mean(np.square(residuals.flatten())))
    #     # print(pred2[i].dtype)
    #     # print(ex_y_valid[i].dtype)
    #     # print("residuals")
    #     # print(residuals)
    #     # print("prediction")
    #     # print(pred2[i])
    #     # print("true y")
    #     # print(ex_y_valid[i])
    #
    #
    #
    #     plt.figure(figsize=(8 ,8))
    #     plt.imshow(ex_x_valid[i])
    #     plt.figure(figsize=(4 ,4)).canvas.set_window_title("Target")
    #     plt.imshow(ex_y_valid[i])
    #
    #     ran_res = np.max(residuals) - np.min(residuals)
    #     if ran_res != 0:
    #         plt.figure(figsize=(4 ,4))
    #         plt.imshow((residuals - np.min(residuals) ) /ran_res)
    #     else:
    #         print("All residuals are 0")
    #
    #
    #
    #
    #     plt.figure(figsize=(4 ,4)).canvas.set_window_title("prediction")
    #     plt.imshow(pred2[i])
    #
    #
    #     plt.show()
