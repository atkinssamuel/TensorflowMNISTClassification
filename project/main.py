from data_management.data_manipulation import import_MNIST, load_MNIST, load_MNIST_flat
from project.dense.dense_train import dense_train
from project.dense.dense_test import dense_test
from project.conv.conv_train import conv_train
from project.conv.conv_test import conv_test


def dense(_train, x_train, y_train, x_test, y_test):
    # Training Parameters:
    learning_rate = 0.001
    num_epochs = 50
    num_models = 50
    batch_size = 512
    checkpoint_frequency = 2
    # Testing Parameters:
    checkpoint_file = "conv_epoch_40.ckpt"
    if _train:
        dense_train(x_train, y_train, learning_rate, num_epochs, batch_size, checkpoint_frequency=checkpoint_frequency,
                    num_models=num_models)
    else:
        dense_test(x_test, y_test, checkpoint_file)


def conv(_train, x_train, y_train, x_test, y_test):

    # Training Parameters:
    learning_rate = 0.0001
    num_epochs = 50
    num_models = 100
    batch_size = 512
    checkpoint_frequency = 2
    # Testing Parameters:
    checkpoint_file = "conv_epoch_48.ckpt"
    if _train:
        conv_train(x_train, y_train, learning_rate, num_epochs, batch_size, checkpoint_frequency=checkpoint_frequency,
                    num_models=num_models)
    else:
        conv_test(x_test, y_test, checkpoint_file)


if __name__ == "__main__":
    _train = 1
    (x_train, y_train), (x_test, y_test) = load_MNIST()
    (x_train, y_train), (x_test, y_test) = (x_train, y_train), (x_test, y_test)
    conv(_train, x_train, y_train, x_test, y_test)

