import tensorflow as tf
import numpy as np
#from tensorflow.examples.tutorials.mnist import input_data
#mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

import pickle
import notMNIST_data as notMNIST

#TODO: the cost function seems to get stuck at 2.303 (a local minimum)
#       the problem seems to be the network itself, giving it the MNSIT database causes it to fall into the same 2.303 cost trap
#       alternating between relu and sigmoid layers seems to work pretty well.  Accuracy on MNIST training set was around 93
#       after testing on notMNIST dataset: 93% accuracy
#       NOTE: switching sigmoid function to tanh yielded 95.11% accuracy
#       NOTE: to increase the training accuracy, try reading more about neural networks and nonlinearity, there should be a chapter on that in one of your books


#Description: This is a fully_connected neural network with an input layer, two hidden layers, and an output layer
#               The network uses gradient descent optimization with an exponentially decaying learning rate
#               The network also has l2_optimization for the weights in each layer.


output_count = 10
node_count = 300
batch_size = 100
l2_beta_value = 0.01
epochs = 10

def weight_variable(shape):
    #keeping the seed constant for TESTING
    op = tf.Variable(tf.random_normal(shape, stddev=0.03, mean=0.0, seed=55))
    return op
def bias_variable(length):
    #keeping the seed constant for TESTING
    op = tf.Variable(tf.random_normal(length, seed=25))
    return op
def nn_layer(input_tensor, input_dim, output_dim, layer_name, act=tf.nn.relu):
    with tf.name_scope(layer_name):
        with tf.name_scope('weights'):
            weights = weight_variable([input_dim, output_dim])
            tf.summary.histogram(layer_name + "_weights", weights)
        with tf.name_scope('biases'):
            biases = bias_variable([output_dim])
            tf.summary.histogram(layer_name + "_biases", biases)
        with tf.name_scope("Wx_plus_b"):
            preactivate = tf.matmul(input_tensor, weights) + biases
        with tf.name_scope("l2_loss_calculation"):
            #this section computes the l2_loss of the layer based on the weights
            l2 = tf.nn.l2_loss(weights)
            l2 = l2 * l2_beta_value
            tf.summary.scalar(layer_name + "_l2_loss", l2)

        #variables that change throughout training can be tracked with a histogram
        activation = act(preactivate, name='activation')
        tf.summary.histogram("activation", activation)
        return activation, l2

with tf.name_scope("input_data"):
    x = tf.placeholder(tf.float32, [None, 784], name='input_x')
    y = tf.placeholder(tf.float32, [None, 10], name='output_labels')

with tf.name_scope("network_function"):
    input_layer, input_l2 = nn_layer(x, 784, node_count, 'input_layer', act=tf.nn.relu)
    hidden_1, hidden_1_l2 = nn_layer(input_layer, node_count, node_count, 'hidden_1', act=tf.tanh)
    hidden_2, hidden_2_l2 = nn_layer(hidden_1, node_count, node_count, 'hidden_2', act=tf.nn.relu)
    output_layer, output_l2 = nn_layer(hidden_2, node_count, output_count, 'output_layer', act=tf.tanh)
with tf.name_scope("cost_function"):
    cross_entrop = tf.nn.softmax_cross_entropy_with_logits(logits=output_layer, labels=y)
    cross_entrop = tf.reduce_mean(cross_entrop)

    #l2_sum = tf.add_n([input_l2, hidden_2_l2])#tf.add_n is the method used to add more than 2 tensors at once
    cost = tf.reduce_mean(cross_entrop)

with tf.name_scope("optimization"):
    #optimization is with gradient GradientDescentOptimizer
    #the learning_rate of the optimizer will also decay exponentially
    
    global_step = tf.Variable(0, trainable=False)
    start_learning_rate = 0.5
    decay_rate = 0.96
    learning_rate = tf.train.exponential_decay(start_learning_rate, global_step, 100000, decay_rate, staircase=True)
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost, global_step=global_step)

with tf.name_scope('accuracy'):
    correct_pred = tf.equal(tf.argmax(output_layer, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    tf.summary.scalar('accuracy', accuracy)

nmnst_set = notMNIST.dataset()
batch_x, batch_y = nmnst_set.get_train_batches(batch_size)

test_x, test_y = nmnst_set.get_test_dataset()

init_op = tf.global_variables_initializer()
with tf.Session() as sesh:
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter('logs/train/notMNIST/1',
                                          sesh.graph)

    #total_batch = int(len(mnist.train.labels) / batch_size)

    sesh.run(init_op)
    count = 0
    for epoch in range(epochs):
        for i in range(len(batch_y)):

            train_x = batch_x[i]
            train_y = batch_y[i]

            #train_x, train_y = mnist.train.next_batch(batch_size=batch_size)
            print("--------training_day-----------")
            print(train_x.shape)
            summ, _, acc, c = sesh.run([merged, train_step, accuracy, cost], feed_dict={x: train_x, y: train_y})
            train_writer.add_summary(summ, count)
            print("accuracy: " + str(acc))
            print("Epoch:", (epoch + 1), "cost =", "{:.3f}".format(c))
            count += 1
        test_accuracy = sesh.run([accuracy], feed_dict={x: test_x, y: test_y})
        print(test_accuracy)

       

