import numpy as np

class Network:
    def __init__(self, input_arr, output_arr, layers=2):
        #layers is the number of layers in the network
        self.input_dimensions = input_arr.shape
        self.output_dimensions = output_arr.shape
        self.input_arr = input_arr
        self.output_arr = output_arr
        self.layer_count = layers
        self.create_network_arrays(self.input_dimensions, self.output_dimensions, layers)
    def sigmoid(self, x, deriv=False):
        if(deriv == True):
            return x*(1-x)
        return 1/(1 + np.exp(-x))
    def training_input(self, tr_input):
        self.training_data = tr_input
    def create_network_arrays(self, input_dim, output_dim, layers):
        array = []
        for i in range(layers):
            item = None
            if i == 0:
                item = 2 * np.random.random((input_dim[1],output_dim[0])) - 1
            elif i == layers - 1:
                item = 2 * np.random.random((output_dim[0],output_dim[1])) - 1
            else:
                item = 2 * np.random.random((output_dim[0],output_dim[0])) - 1
            array.append(item)
        self.network_array = array
    def activate_network(self, data_input):
        l0 = data_input
        self.l_data = []
        self.l_data.append(l0)
        for a in range(self.layer_count):
            l0 = self.sigmoid(np.dot(l0, self.network_array[a]))
            self.l_data.append(l0)
        return l0
    def train(self, epoch_number):
        for not_important in range(epoch_number):
            #what does the train function have to do?
            #it has to go backward through each layer, training each layer for the answers
            weight_update = [None] * self.layer_count
            error = self.output_arr - self.activate_network(self.input_arr)
            for i in range(self.layer_count - 1, -1, -1):
                delta = error * self.sigmoid(self.l_data[i + 1], True)
                error = delta.dot(self.network_array[i].T)
                weight_update[i] = np.dot(self.l_data[i].T, delta)

            for i in range(len(weight_update)):
                self.network_array[i] += weight_update[i]




x = np.array([[0,1], [1,0], [0,0], [1,1]])
y = np.array([[0,0,1,1]])
a = Network(x, y.T, layers=5)
a.train(10000)
