from data_management.data_manipulation import import_MNIST, load_MNIST, load_MNIST_flat
from project.dense.dense_train import dense_train
from project.dense.dense_test import dense_test
from project.conv.conv_train import conv_train
from project.conv.conv_test import conv_test


if __name__ == "__main__":
    _train = 1

    (x_train, y_train), (x_test, y_test) = load_MNIST_flat()

    # Training Parameters:
    learning_rate = 0.001
    num_epochs = 500
    num_models = 10
    batch_size = 64
    # Testing Parameters:
    checkpoint_file = "epoch_90.ckpt"
    if _train:
        dense_train(x_train, y_train, learning_rate, num_epochs, batch_size, checkpoint_frequency=10,
                    num_models=num_models)
    else:
        dense_test(x_test, y_test, checkpoint_file)
