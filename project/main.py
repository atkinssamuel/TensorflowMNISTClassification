from data_management.data_manipulation import import_MNIST, load_MNIST, load_MNIST_flat
from project.train import train


if __name__ == "__main__":
    _train = 1

    (x_train, y_train), (x_test, y_test) = load_MNIST_flat()

    # Training Parameters:
    learning_rate = 0.001
    num_epochs = 100
    num_models = 10
    batch_size = 64
    # Testing Parameters:
    checkpoint_file = "epoch_1990.ckpt"
    if _train:
        train(x_train, y_train, learning_rate, num_epochs, batch_size, checkpoint_frequency=10, num_models=num_models)
