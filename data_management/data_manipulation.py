from mnist import MNIST
import numpy as np
import matplotlib.pyplot as plt

MNIST_dir = "data_management/dataset/"
np_save_dir = "data_management/np_dataset/"


def import_MNIST():
    # Ensure you replace the dots in the dataset files with "-"s to avoid a "File not found" error
    MNIST_data = MNIST(MNIST_dir)

    x_train, y_train = MNIST_data.load_training()
    x_test, y_test = MNIST_data.load_testing()

    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_test = np.array(x_test)
    y_test = np.array(y_test)

    np.save(np_save_dir + 'x_train', x_train)
    np.save(np_save_dir + 'y_train', y_train)
    np.save(np_save_dir + 'x_test', x_test)
    np.save(np_save_dir + 'y_test', y_test)
    return


def load_MNIST():
    image_dim = 28

    x_train = np.load(np_save_dir + "x_train.npy")
    y_train = np.load(np_save_dir + "y_train.npy")
    x_test = np.load(np_save_dir + "x_test.npy")
    y_test = np.load(np_save_dir + "y_test.npy")

    x_train = x_train.reshape((np.shape(x_train)[0], image_dim, image_dim))
    x_test = x_test.reshape((np.shape(x_test)[0], image_dim, image_dim))

    return (x_train, y_train), (x_test, y_test)


def load_MNIST_flat():
    x_train = np.load(np_save_dir + "x_train.npy")
    y_train = np.load(np_save_dir + "y_train.npy")
    x_test = np.load(np_save_dir + "x_test.npy")
    y_test = np.load(np_save_dir + "y_test.npy")

    return (x_train, y_train), (x_test, y_test)

