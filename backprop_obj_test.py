import numpy as np
import backprop_objects as bp

'''x = np.array([[0,0,1], [2,2,1], [1,0,1], [1,1,1], [2,3,1]])
y = np.array([[1],[1],[0],[1],[0]])'''

pre_x = [[6.2,2.2,4.5,1.5],
         [4.6,3.1,1.5,0.2],
         [7.9,3.8,6.4,2.0]]
pre_y = [[1],
         [0],
         [2]]
x = np.array(pre_x)
y = np.array(pre_y)

net = bp.Network(x, y, layers=3)
print(x)
print(y)
net.train(10000)
print(net.activate_network(x[1]))
