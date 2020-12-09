# Rectified Linear object
import numpy as np

np.random.seed(0)

X = [[1, 2, 3, 2.5],
     [2.0, 5.0, -1.0, 2.0],
     [-1.5, 2.7, 3.3, -0.8]]

class Layer_Dense: # create class object
    def __init__(self,n_inputs, n_neurons):
        # these are our weights
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons) # this is our shape
        # these are our biases
        self.biases = np.zeros((1, n_neurons)) # shape is 1 by n neurons we have
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

class Activation_ReLu:
    # forward method for the rectified linear activation function
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)

layer1 = Layer_Dense(4,5)
layer2 = Layer_Dense(5,2)
