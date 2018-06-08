import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
#TODO: the training accuracy is really low, I am relativly sure that the

learning_rate = 0.5
epochs = 5
batch_size = 10 #working with a smaller batch size actually improves the accuracy from ~95 to 97.07

with tf.name_scope('data_input'):
    x = tf.placeholder(tf.float32, [None, 784], name='training_input')
    y = tf.placeholder(tf.float32, [None, 10], name='training_labels')

def weight_variable(shape):
    #keeping the seed constant for TESTING
    op = tf.Variable(tf.random_normal(shape, stddev=0.03, mean=0.0, seed=55))
    return op
def bias_variable(length):
    #keeping the seed constant for TESTING
    op = tf.Variable(tf.random_normal(length, seed=25))
    return op
def nn_layer(input_tensor, input_dim, output_dim, layer_name, act=tf.nn.relu):
    #it seems that the sigmoid activation function works much better than the RELU one
    with tf.name_scope(layer_name):
        with tf.name_scope('weights'):
            weights = weight_variable([input_dim, output_dim])
            tf.summary.histogram(layer_name + "_weights", weights)
        with tf.name_scope('biases'):
            biases = bias_variable([output_dim])
            tf.summary.histogram(layer_name + "_biases", biases)
        with tf.name_scope("Wx_plus_b"):
            preactivate = tf.matmul(input_tensor, weights) + biases
        #variables that change throughout training can be tracked with a histogram
        activation = act(preactivate, name='activation')
        tf.summary.histogram("activation", activation)
        return activation
#the first layer takes in 784 pixels (28**2) and has 500 nodes
with tf.name_scope('computation'):
    hidden_1 = nn_layer(x, 784, 300, 'layer1', tf.nn.relu)
    #hidden_2 = nn_layer(hidden_1, 300, 300, 'layer2', tf.sigmoid)
    keep_prob = tf.placeholder(tf.float32)
    dropped = tf.nn.dropout(hidden_1, keep_prob)

    output = nn_layer(dropped, 300, 10, 'output_layer', tf.sigmoid)


with tf.name_scope('softmax_cross-entropy'):
    #maybe try a new loss function
    out_soft = tf.nn.softmax(output)
    y_clipped = tf.clip_by_value(out_soft, 1e-10, 0.9999999)
    cross_entropy = -tf.reduce_mean(tf.reduce_sum(y * tf.log(y_clipped)
                         + (1 - y) * tf.log(1 - y_clipped), axis=1))

    #cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=y))
    tf.summary.scalar("cost_1", cross_entropy)
with tf.name_scope('train'):
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)

with tf.name_scope('accuracy'):
    correct_pred = tf.equal(tf.argmax(output, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    tf.summary.scalar('accuracy', accuracy)
init_op = tf.global_variables_initializer()
with tf.Session() as sesh:
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter('logs/train',
                                          sesh.graph)
    sesh.run(init_op)
    total_batch = int(len(mnist.train.labels) / batch_size)
    count = 0
    for epoch in range(epochs):
        for i in range(total_batch):
            batch_x, batch_y = mnist.train.next_batch(batch_size=batch_size)
            summ, _, c, acc = sesh.run([merged, train_step, cross_entropy, accuracy],
                         feed_dict={x: batch_x, y: batch_y, keep_prob: 0.8})
            train_writer.add_summary(summ, count)
            print("accuracy: " + str(acc))
            print("Epoch:", (epoch + 1), "cost =", "{:.3f}".format(c))
            count += 1
        test_accuracy = sesh.run([accuracy], feed_dict={x: mnist.test.images, y: mnist.test.labels, keep_prob: 1.0})
        print(test_accuracy)
