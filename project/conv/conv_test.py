import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

conv_checkpoint_dir = "weights/conv/"
conv_results_folder = "results/conv/"


def conv_test(x_test, y_test, checkpoint_file):
    # Parameters:
    # Base Params:
    categories = 10
    image_dim = 28
    input_depth = 1
    # Conv Layer 1:
    # Input: None x 28 x 28 x 1
    layer_1_depth = 6
    k1_width = 5
    stride_1 = 1
    padding_1 = "SAME"
    # Conv Layer 2:
    # Input: None x 28 x 28 x 6
    layer_2_depth = 18
    k2_width = 7
    stride_2 = 1
    padding_2 = "SAME"
    # Output: None x 28 x 28 x 18
    # Pooling Layer 2:
    # Output: None x 14 x 14 x 18
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

    # For weight saving:
    saver = tf.train.Saver()
    checkpoint = conv_checkpoint_dir + checkpoint_file

    with tf.Session() as sess:
        saver.restore(sess, checkpoint)

        test_output = sess.run(y, feed_dict={x: x_test, y_: y_test})
        predictions = np.argmax(test_output, axis=1)
        targets = np.argmax(y_test, axis=1)
        accuracy = np.sum(np.equal(predictions, targets))/test_output.shape[0] * 100

        print(f"Testing Accuracy = {accuracy}%")

    return
