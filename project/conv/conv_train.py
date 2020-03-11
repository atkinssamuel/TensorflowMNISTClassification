import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

conv_checkpoint_dir = "weights/conv/"
results_folder = "results/conv/"



def cross_entropy(p, q):
	return -sum([p[i] * np.log(q[i]) for i in range(len(p))])


def conv_train(x_train, y_train, learning_rate, num_epochs, batch_size, checkpoint_frequency=10, num_models=200):
    # Parameters:
    # Base Params:
    categories = 10
    image_dim = 28
    input_depth = 1
    # Conv-Pooling Layer 1:
    # Input: None x 28 x 28 x 1
    layer_1_depth = 6
    k1_width = 5
    stride_1 = 1
    padding_1 = "SAME"
    # Conv-Pooling Layer 2:
    # Input: None x 14 x 14 x 6
    layer_2_depth = 18
    k2_width = 7
    stride_2 = 1
    padding_2 = "SAME"
    # Output: None x 7 x 7 x 18
    conv_output_nodes = 7 * 7 * 18

    # Fully Connected:
    increase_factor = 1.5
    hidden_layer_3 = round(conv_output_nodes * increase_factor)
    hidden_layer_4 = round(hidden_layer_3 * increase_factor)
    output_layer = 10

    # Defining Layers:
    # Defining Placeholders:
    x = tf.placeholder(tf.float32, [None, image_dim, image_dim, input_depth])
    y_ = tf.placeholder(tf.float32, [None, categories])

    # [filter_height, filter_width, in_channels, out_channels]
    # Conv-Pool:
    k1 = tf.Variable(tf.random_normal([k1_width, k1_width, input_depth, layer_1_depth]))
    y1 = tf.nn.leaky_relu(tf.nn.conv2d(x, k1, [stride_1], padding_1))
    y1 = tf.nn.pool(y1, [2, 2], pooling_type="MAX", padding="SAME", strides=[2, 2])
    # Conv-Pool:
    k2 = tf.Variable(tf.random_normal([k2_width, k2_width, layer_1_depth, layer_2_depth]))
    y2 = tf.nn.leaky_relu(tf.nn.conv2d(y1, k2, [stride_2], padding_2))
    y2 = tf.nn.pool(y2, [2, 2], pooling_type="MAX", padding="SAME", strides=[2, 2])
    y2 = tf.reshape(y2, [-1, conv_output_nodes])

    # Layer 3 variables:
    W3 = tf.Variable(tf.truncated_normal([conv_output_nodes, hidden_layer_3], stddev=0.15))
    b3 = tf.Variable(tf.zeros([hidden_layer_3]))
    y3 = tf.math.sigmoid(tf.matmul(y2, W3) + b3)
    # Layer 4 variables:
    W4 = tf.Variable(tf.truncated_normal([hidden_layer_3, hidden_layer_4], stddev=0.15))
    b4 = tf.Variable(tf.zeros([hidden_layer_4]))
    y4 = tf.math.sigmoid(tf.matmul(y3, W4) + b4)
    # Layer 5 Variables:
    W5 = tf.Variable(tf.truncated_normal([hidden_layer_4, output_layer], stddev=0.15))
    b5 = tf.Variable(tf.zeros([output_layer]))
    y = tf.matmul(y4, W5) + b5

    cost = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y)
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

    # Miscellaneous quantities:
    sample_count = np.shape(x_train)[0]

    # For weight saving:
    saver = tf.train.Saver(max_to_keep=num_models)

    training_losses = []
    training_accuracies = []

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch_iteration in range(num_epochs):
            for batch in range(int(sample_count / batch_size)):
                batch_x = x_train[batch * batch_size: (1 + batch) * batch_size]
                batch_y = y_train[batch * batch_size: (1 + batch) * batch_size]
                # Instantiating the inputs and targets with the batch values:
                output = np.array(sess.run([optimizer], feed_dict={x: batch_x, y_: batch_y}))
            training_output, training_loss = sess.run([y, cost], feed_dict={x: x_train, y_: y_train})
            training_loss = np.mean(training_loss)
            training_losses.append(training_loss)
            training_predictions = np.argmax(training_output, axis=1)
            training_targets = np.argmax(y_train, axis=1)
            training_accuracy = round(np.sum(np.equal(training_predictions, training_targets)) \
                                / training_predictions.shape[0] * 100, 2)
            training_accuracies.append(training_accuracy)
            print(f"Current Epoch = {epoch_iteration}, Training Loss = {training_loss}, "
                  f"Training Accuracy = {training_accuracy}%, {round(epoch_iteration / num_epochs * 100, 2)}% Complete")

            if epoch_iteration % checkpoint_frequency == 0:
                checkpoint = conv_checkpoint_dir + f"conv_epoch_{epoch_iteration}.ckpt"
                saver.save(sess, checkpoint)
        sess.close()

    # Loss Plotting:
    plt.title("Training Loss:")
    plt.ylabel("Loss")
    plt.xlabel("Epoch Iteration")
    plt.plot(training_losses)
    plt.savefig(results_folder + "training_loss.png")
    plt.show()
    # Accuracy Plotting:
    plt.title("Training Accuracy:")
    plt.ylabel("Accuracy %")
    plt.xlabel("Epoch Iteration")
    plt.plot(training_accuracies, "g")
    plt.savefig(results_folder + "training_accuracy.png")
    plt.show()
    return
