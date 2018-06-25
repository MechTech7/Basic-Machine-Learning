import tensorflow as tf
import numpy as np


from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

graph = tf.Graph()
batch_size = 50

conv_padding = "VALID"
#NOTE: convolutional neural nets are much better at the MNIST problem, we reached 98% accuracy on MNIST test set as opposed to around 93 with full connected neural nets
#NOTE: tensors have 4 dimensions [batch, height, width, channel]
with graph.as_default():
	def generate_weights_biases(weights_shape, biases_shape, name):
		with tf.name_scope("generate_" + name):
			weights = tf.Variable(tf.truncated_normal(weights_shape, stddev=0.1), name="weights_" + name)
			biases = tf.Variable(tf.zeros(biases_shape), name="biases_" + name)
			return weights, biases

	with tf.name_scope("data"):
		x = tf.placeholder(tf.float32, [batch_size, 28, 28, 1], name="train_data")
		y = tf.placeholder(tf.float32, [batch_size, 10], name="train_labels")
		keep_prob = tf.placeholder(tf.float32, name="keep_prob")

	with tf.name_scope("conv_layer_1"):
		#NOTE: figure out how weights interact with 28x28 images
		#the depth of the output volume is a hyperparameter it corresponds to the number of filters we use
		weights_conv1, biases_conv1 = generate_weights_biases([5, 5, 1, 16], [16], "conv_1")

		#convolves across the image and computes the activation function
		#the [1, 1, 1, 1] stride vector means that the convolution moves across 1 batch, 1 pixel wide, 1 pixel high, and 1 pixel in each channel
		
		#the stride for batch and channel should always be kept at 1 so that examples and channels are not skipped
		logits = tf.nn.conv2d(x, weights_conv1, strides=[1, 1, 1, 1], padding=conv_padding) + biases_conv1
		logits = tf.nn.relu(logits)
		print("conv_1_activation_size: " + str(logits.shape))
		with tf.name_scope("max_pool_1"):
			logits = tf.nn.max_pool(logits, strides=[1, 2, 2, 1], ksize=[1, 2, 2, 1], padding="SAME")

		print("conv_1_output_size: " + str(logits.shape))
	with tf.name_scope("conv_layer_2"):
		weights_conv2, biases_conv2 = generate_weights_biases([5, 5, 16, 16], [16], "conv2")
		# Second convolutional layer for the training data
		logits = tf.nn.conv2d(logits, weights_conv2, strides=[1, 1, 1, 1], padding=conv_padding) + biases_conv2
		logits = tf.nn.relu(logits)
		# Second convolutional layer fot the test data
		print("conv_2_activation_size: " + str(logits.shape))
		with tf.name_scope("max_pool_2"):
			# Pooling for training and testing data
			logits = tf.nn.max_pool(logits, strides=[1, 2, 2, 1], ksize=[1, 2, 2, 1], padding='SAME')
			
		print("conv_2_output_size: " + str(logits.shape))
	with tf.name_scope("connected_1"):
		weights_1, biases_1 = generate_weights_biases([7*7*16, 1024], [1024], "fully1")
		# Reshape the training logits to be a two-dimensional tensor
		logits = tf.reshape(logits, [-1, 7*7*16])
		# Perform matrix multiplication and add a relu layer
		logits = tf.nn.relu(tf.matmul(logits, weights_1) + biases_1, name="relu1")
		# Apply dropout
		logits = tf.nn.dropout(logits, keep_prob=keep_prob)
		# Repeat everything for the test data
		print("connected_1_output_size: " + str(logits.shape))
	with tf.name_scope("connected_2"):
		weights_2, biases_2 = generate_weights_biases([1024, 10], [10], "fully2")
		# The second fully connected layer is the last layer, so
		# we do not need to add dropout or a relu layer
		logits = tf.matmul(logits, weights_2) + biases_2
		# Final test dataset matrix multiplication
		print("connnected_2_output_size: " + str(logits.shape))

	with tf.name_scope("loss"):
		# Compute the softmax loss
		loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))
	tf.summary.scalar("loss", loss)

	with tf.name_scope("optimizer"):
		optimizer = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

	with tf.name_scope("accuracy"):
		def calculate_accuracy(log, lab):
			# Calculate the accuracy of our prediction
			# The accuracy is expressed in percentages
			correct_prediction = tf.equal(tf.argmax(log, 1), tf.argmax(lab, 1))
			return tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

		accuracy = calculate_accuracy(logits, y)
	tf.summary.scalar("accuracy", accuracy)
	merged = tf.summary.merge_all()

steps = 4001

with tf.Session(graph=graph) as sess:
	tf.global_variables_initializer().run()
	writer = tf.summary.FileWriter("./logs/conv_mnist/1", tf.get_default_graph())
	print('Initialized!')
	for step in range(steps):
		# Retrieve a batch of MNIST images
		batch_x, batch_y = mnist.train.next_batch(batch_size)
		batch_x = np.reshape(batch_x, [-1, 28, 28, 1])
		# Get a batch of test data
		test_x, test_y = mnist.test.next_batch(batch_size)
		test_x = np.reshape(test_x, [-1, 28, 28, 1])
		# Feed dictionary for the placeholders
		feed_dict = {x: batch_x, y: batch_y, keep_prob: 0.5}

		_, a, l, m = sess.run([optimizer, accuracy, loss, merged], feed_dict=feed_dict)

		# Write to Tensorboard
		writer.add_summary(m, step)

		# Print accuracy and loss
		acc = sess.run([accuracy], feed_dict={x: test_x, y: test_y, keep_prob: 1.0})
		if step % 500 == 0:
			print("Accuracy at step {}: {}".format(step, a))
			print("     loss: {}".format(l))
	# Close the writer
	writer.close()