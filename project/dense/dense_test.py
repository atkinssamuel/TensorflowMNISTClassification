import tensorflow as tf
import numpy as np
from project.dense.dense_train import dense_checkpoint_dir, dense_results_dir


def dense_test(x_test, y_test, checkpoint_file):
    # Parameters:
    input_nodes = np.shape(x_test)[1]
    hidden_layer_1 = 32
    hidden_layer_2 = 64
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
    y2 = tf.matmul(y1, W2) + b2
    # Layer 3 variables:
    W3 = tf.Variable(tf.truncated_normal([hidden_layer_2, output_layer], stddev=0.15))
    b3 = tf.Variable(tf.zeros([output_layer]))
    y = tf.matmul(y2, W3) + b3
    # Placeholder for batch of targets:
    y_ = tf.placeholder(tf.float32, [None, output_layer])

    # For weight saving:
    saver = tf.train.Saver()
    checkpoint = dense_checkpoint_dir + checkpoint_file

    with tf.Session() as sess:
        saver.restore(sess, checkpoint)

        test_output = sess.run(y, feed_dict={x: x_test, y_: y_test})
        predictions = np.argmax(test_output, axis=1)
        targets = np.argmax(y_test, axis=1)
        accuracy = np.sum(np.equal(predictions, targets))/test_output.shape[0] * 100

        print(f"Testing Accuracy = {accuracy}%")


    return