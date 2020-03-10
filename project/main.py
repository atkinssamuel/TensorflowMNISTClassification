from data_management.data_manipulation import import_MNIST, load_MNIST, load_MNIST_flat
from project.dense.dense_train import dense_train
from project.dense.dense_test import dense_test
from project.convolutional.convolutional_train import convolutional_train
from project.convolutional.convolutional_test import convolutional_test


if __name__ == "__main__":
    _train = 0

    (x_train, y_train), (x_test, y_test) = load_MNIST_flat()

    # Training Parameters:
    learning_rate = 0.001
    num_epochs = 500
    num_models = 10
    batch_size = 64
    # Testing Parameters:
    checkpoint_file = "epoch_90.ckpt"
    if _train:
        train(x_train, y_train, learning_rate, num_epochs, batch_size, checkpoint_frequency=10, num_models=num_models)
    else:
        test(x_test, y_test, checkpoint_file)
