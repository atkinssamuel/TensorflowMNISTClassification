from data_management.data_manipulation import import_MNIST, load_MNIST


if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test) = load_MNIST()
