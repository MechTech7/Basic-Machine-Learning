import notMNIST_downloads as notMNIST
import numpy as np
import cv2
import random

class dataset():
	def __init__(self):
		self.global_step = 0

		train_folders = ['./notMNIST_large/A', './notMNIST_large/B', './notMNIST_large/C', './notMNIST_large/D', './notMNIST_large/E', './notMNIST_large/F', './notMNIST_large/G', './notMNIST_large/H', './notMNIST_large/I', './notMNIST_large/J']
		test_folders = ['./notMNIST_small/A', './notMNIST_small/B', './notMNIST_small/C', './notMNIST_small/D', './notMNIST_small/E', './notMNIST_small/F', './notMNIST_small/G', './notMNIST_small/H', './notMNIST_small/I', './notMNIST_small/J']

		self.train_datasets = notMNIST.maybe_pickle(train_folders, 45000)
		self.test_datasets = notMNIST.maybe_pickle(test_folders, 1800)

		train_size = 200000
		valid_size = 10000
		test_size = 10000

		self.valid_dataset, self.valid_labels, self.train_dataset, self.train_labels = notMNIST.merge_datasets(self.train_datasets, train_size, valid_size)
		_, _, self.test_dataset, self.test_labels = notMNIST.merge_datasets(self.test_datasets, test_size)


	def randomize(self, dataset, labels):
		permutation = np.random.permutation(labels.shape[0])
		shuffled_dataset = dataset[permutation,:,:]
		shuffled_labels = labels[permutation]
		return shuffled_dataset, shuffled_labels
	def convert_to_one_hot(self, input_arr, depth):
		#converts an input numpy array of integers into a numpy array of one_hot values
		length = input_arr.shape[0]
		op = np.zeros((length, depth))
		op[np.arange(length), input_arr] = 1
		return op
	def get_train_dataset(self, train_dataset, train_labels):
		#ERROR: local variable 'train_dataset' referenced before assignment
		train_dataset, train_labels = self.randomize(train_dataset, train_labels)
		train_dataset = train_dataset.flatten().reshape(train_dataset.shape[0], 784)#this flattens each 28 x 28 image into a 1-D array with 784 elements

		train_labels = self.convert_to_one_hot(train_labels, 10)
		return train_dataset, train_labels
	
	def get_test_dataset(self):
		return self.get_train_dataset(self.test_dataset, self.test_labels)
	def get_test_image_dataset(self):
		test_data = self.test_dataset.flatten().reshape(self.test_dataset.shape[0], 28, 28, 1)
		test_l = self.convert_to_one_hot(self.test_labels, 10)

		return test_data, test_l
	def show_random_training(self):
		index = random.randint(0, len(self.train_dataset) - 1)
		img = self.train_dataset[index]

		cv2.imshow('this is it', img)
		cv2.waitKey(0)
		cv2.destroyAllWindows()
	def get_train_image_batches(self, batch_size):
		#returns the training data as 28x28 images
		train_data, train_l = self.train_dataset, self.train_labels
		train_data = train_data.flatten().reshape(train_data.shape[0], 28, 28, 1)

		train_l = train_labels = self.convert_to_one_hot(train_l, 10)

		remainder = train_data.shape[0] % batch_size
		frac = int(train_data.shape[0] / batch_size)

		output_data = []
		output_labels = []

		for i in range(frac):
			start = i*batch_size
			end = (i+1)*batch_size
			output_data.append(train_data[start:end])#this is a numpy array
			output_labels.append(train_l[start:end])
		if remainder != 0:
			#this cleans up in case there is an uneven division of batches
			all_end = frac*batch_size
			output_data.append(train_data[all_end:all_end + remainder])
			output_labels.append(train_l[all_end:all_end + remainder])
		return output_data, output_labels
	def get_train_batches(self, batch_size):
		train_data, train_l = self.get_train_dataset(self.train_dataset, self.train_labels)
		remainder = train_data.shape[0] % batch_size
		frac = int(train_data.shape[0] / batch_size)

		output_data = []
		output_labels = []

		for i in range(frac):
			start = i*batch_size
			end = (i+1)*batch_size
			output_data.append(train_data[start:end])#this is a numpy array
			output_labels.append(train_l[start:end])
		if remainder != 0:
			#this cleans up in case there is an uneven division of batches
			all_end = frac*batch_size
			output_data.append(train_data[all_end:all_end + remainder])
			output_labels.append(train_l[all_end:all_end + remainder])
		return output_data, output_labels

