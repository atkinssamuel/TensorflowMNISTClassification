import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

dense_checkpoint_dir = "weights/dense/"
dense_results_dir = "results/dense/"


def dense_train(x_train, y_train, learning_rate, num_epochs, batch_size, checkpoint_frequency=10, num_models=200):
    # Parameters:
    input_nodes = np.shape(x_train)[1]

    # Fully Connected:
    increase_factor = 1.5
    hidden_layer_1 = round(input_nodes * increase_factor)
    hidden_layer_2 = round(hidden_layer_1 * increase_factor)
    output_layer = 10


    # Defining Layers:
    # Placeholder for batch of inputs:
    x = tf.placeholder(tf.float32, [None, input_nodes])
    # Layer 1 variables:
    W1 = tf.Variable(tf.truncated_normal([input_nodes, hidden_layer_1], stddev=0.15))
    b1 = tf.Variable(tf.zeros([hidden_layer_1]))
    y1 = tf.math.sigmoid(tf.matmul(x, W1) + b1)
    # Layer 2 variables:
    W2 = tf.Variable(tf.truncated_normal([hidden_layer_1, hidden_layer_2], stddev=0.15))
    b2 = tf.Variable(tf.zeros([hidden_layer_2]))
    y2 = tf.math.sigmoid(tf.matmul(y1, W2) + b2)
    # Layer 3 variables:
    W3 = tf.Variable(tf.truncated_normal([hidden_layer_2, output_layer], stddev=0.15))
    b3 = tf.Variable(tf.zeros([output_layer]))
    y = tf.matmul(y2, W3) + b3
    # Placeholder for batch of targets:
    y_ = tf.placeholder(tf.float32, [None, output_layer])

    cost = tf.reduce_sum(tf.math.square(y - y_))
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

    # Miscellaneous quantities:
    sample_count = np.shape(x_train)[0]

    # For weight saving:
    saver = tf.train.Saver(max_to_keep=num_models)

    training_losses = []

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch_iteration in range(num_epochs):
            for batch in range(int(sample_count / batch_size)):
                batch_x = x_train[batch * batch_size: (1 + batch) * batch_size]
                batch_y = y_train[batch * batch_size: (1 + batch) * batch_size]
                # Instantiating the inputs and targets with the batch values:
                sess.run([optimizer], feed_dict={x: batch_x, y_: batch_y})
            training_output = sess.run([y], feed_dict={x: x_train, y_: y_train})[0]
            training_loss = np.sum(np.square(training_output - y_train) / np.shape(training_output)[0])
            training_losses.append(training_loss)

            if epoch_iteration % checkpoint_frequency == 0:
                checkpoint = dense_checkpoint_dir + f"dense_epoch_{epoch_iteration}.ckpt"
                saver.save(sess, checkpoint)
        sess.close()
    plt.title("Training Loss:")
    plt.ylabel("Loss")
    plt.xlabel("Epoch Iteration")
    plt.plot(training_losses)
    plt.savefig(dense_results_dir + "training_loss.png")
    plt.show()
    return
