import numpy as np
import tensorflow as tf
import cv2
#import notMNIST_data as notMNIST

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

#this will be a convolutional neural net of my own design working on a dataset of my choosing
#we'll start with notMNIST 'cause that's close to MNIST and still more difficult


epoch_count = 1
node_count = 16
batch_size = 50
conv_padding = "SAME"

conn_node_count = 1024
def generate_weights_biases(weights_shape, biases_shape):
		weights = tf.Variable(tf.truncated_normal(weights_shape, stddev=0.1))
		biases = tf.Variable(tf.zeros(biases_shape))
		return weights, biases
def conv_layer(input_data, weights_shape, biases_shape, strides, name):
	with tf.name_scope("convolution_layer_" + name):
		weights, biases = generate_weights_biases(weights_shape, biases_shape)
		linear = tf.nn.conv2d(input_data, weights, strides=strides, padding=conv_padding) + biases
		#NOTE: maybe try tf.nn.elu function as well
		tf.summary.histogram(name + "_weights", weights)
		activation = tf.nn.relu(linear)
	return activation
def max_pool_layer(input_data, strides, ksize, name):
	with tf.name_scope("max_pooling_" + name):
		#NOTE: the ksize window is the window that the max_pool function slides with
		#the strides are the same as convolutional strides
		max_out = tf.nn.max_pool(input_data, ksize=ksize, strides=strides, padding="SAME")
	print(name + " " + str(max_out.shape))
	return max_out
def fully_connected(input_data, weights_shape, biases_shape, activation_func=tf.nn.relu):
	with tf.name_scope("fully_connected_"):
		weights, biases = generate_weights_biases(weights_shape, biases_shape)
		mul = tf.matmul(input_data, weights)
		linear = mul + biases
		activation = activation_func(linear)
	return activation
def no_activation(input_data):
	return input_data

with tf.name_scope("input"):
	x = tf.placeholder(tf.float32, [None, 28, 28, 1], name="x")
	y = tf.placeholder(tf.float32, [None, 10], name="y")
	keep_prob = tf.placeholder(tf.float32, name="keep_prob")

#Note: the network many need some restructuring, feel free to experiment with different types of structures
with tf.name_scope("network"):
	#NOTE: this network is severely overfitting.  Training accuracy is at 100% while test accuracy is at 10%
	#maybe try batch normalization and dropout
	layer_one = conv_layer(x, [5, 5, 1, 16], [16], [1, 1, 1, 1], "one")
	
	max_layer_one = max_pool_layer(layer_one, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], name="one")

	layer_two = conv_layer(max_layer_one, [5, 5, 16, 16], [16], [1, 1, 1, 1], "two")
	#finish up other layers!

	max_pool_two = max_pool_layer(layer_two, [1, 2, 2, 1], [1, 2, 2, 1], "one")
	#the shape coming out of this is (?, 14, 14, 16)

	#convert this to a flattened tensor
	#the [-1] means that that dimension is inferred; its based on the dimension of the batch function
	feature_count = 7 * 7 * 16
	flattened = tf.reshape(max_pool_two, [-1, feature_count])

	with tf.name_scope("full_connected"):
		full_conn_one = fully_connected(flattened, [feature_count, conn_node_count], [conn_node_count], tf.nn.relu)

		#full_conn_two = fully_connected(full_conn_one, [conn_node_count, 50], [50])
		dropout = tf.nn.dropout(full_conn_one, keep_prob=keep_prob)
		output_layer = fully_connected(dropout, [conn_node_count, 10], [10], no_activation)

with tf.name_scope("loss"):
	loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output_layer, labels=y))
	tf.summary.scalar("loss", loss)

with tf.name_scope("accuracy"):
	correct_prediction = tf.equal(tf.argmax(output_layer, 1), tf.argmax(y, 1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	tf.summary.scalar("accuracy", accuracy)


with tf.name_scope("optimizer"):
		train_step = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

#nmnst_set = notMNIST.dataset()
#batch_x, batch_y = nmnst_set.get_train_image_batches(batch_size)

#test_x, test_y = nmnst_set.get_test_image_dataset()

init_op = tf.global_variables_initializer()

with tf.Session() as sesh:
	init_op.run()
	merged = tf.summary.merge_all()
	
	writer = tf.summary.FileWriter("./logs/own_conv_not_mnist/1")
	print("-----------Initialization Complete!------------------")
	count = 0
	epoch = 0
	steps = 2000

	for i in range(steps):
	#for i in range(epoch_count):
		#for j in range(len(batch_y)):

		train_x, train_y = mnist.train.next_batch(batch_size)
		train_x = np.reshape(train_x, [-1, 28, 28, 1])
		# Get a batch of test data
		

		summ, _, acc, c = sesh.run([merged, train_step, accuracy, loss], feed_dict={x: train_x, y: train_y, keep_prob: 0.5})

		writer.add_summary(summ, count)

		if count % 20 == 0:
			print("accuracy: " + str(acc))

		count += 1
	epoch += 1

	test_x, test_y = mnist.test.next_batch(batch_size)
	test_x = np.reshape(test_x, [-1, 28, 28, 1])
	test_accuracy = sesh.run([accuracy], feed_dict={x: test_x, y: test_y, keep_prob: 1.0})
	print("-------------test accuracy: " + str(test_accuracy) + "---------------") 






