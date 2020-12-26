# Rectified Linear object
# pip install nnfs # to get info and code from video examples
import numpy as np

#np.random.seed(0) # for this to work in a notebook all code must run in same cell
import nnfs # instead of random seed, it will also set default data type for numpy
# dot product in numpy will sometime use a different datatype
# there is no way to set default datatype in numpy, it just decides
# with nnfs we overwrite some things to everyone uses same datatype.
# this will make it possible to replicate everything.
# nnfs will also give us some data
# import a dataset from nnfs that is dataset of spirals
from nnfs.datasets import spiral_data

nnfs.init()

X = [[1, 2, 3, 2.5],
     [2.0, 5.0, -1.0, 2.0],
     [-1.5, 2.7, 3.3, -0.8]]

# Create our data. spiral_data creates both features X and labels y
X, y = spiral_data(100, 3) # 100 feature sets of 3 classes

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
        self.output = np.maximum(0, inputs) # alot of info for basically this!

# Next we do one layer for now
layer1 = Layer_Dense(2,5) # first layer from our input data X
# 4 was number of features (size) per sample
# It becomes 2 in our data, since we will create X and y data in a 2D space

# Define a activation function. A ReLu object. Will be our input.
# It will take all of the values from our neurons and produce the activations
# for this entire layer.
activation1 = Activation_ReLu()

layer1.forward(X) #we´ve done inputs*weights+bias, layer1 has a dot output
#print(layer1.output) # prints also some negative values after we done then
# weights and the biases, this should go away after it goes through That
# Rectivies Linear function that should make all negatives 0.
activation1.forward(layer1.output) #layer
print(activation1.output)
# now we have no negative values, many zeros and positive values.
# When the optimizer start optimizing things, those values are going to get
# tweaked over time and we probably will not have as many pure 0 worthless
# values. Also if we see that everthing still goes to 0, the network is dying
# recall we can do to fix this immediately is to initialize biases that are 
# some non-zero number for example. This is unlikely though.

# This should explain why we use simple Rectified Linear activation functions
# Next we will talk about softmax activation function.
