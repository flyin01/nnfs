# Create an object, keeping the input as our sample data delete the rest

import numpy as np

np.random.seed(0)

# Batch of inputs, change name to capital X, standard in ML for features
# We have three samples
X = [[1, 2, 3, 2.5],
     [2.0, 5.0, -1.0, 2.0],
     [-1.5, 2.7, 3.3, -0.8]]

# How to initalize layers.
# 1.Use a trained model, i.e you have already weights and biases. So you sets
# these to what evere they were in the saved model
# 2.Or in our case, we make a new neural network, we have weights and biases
# that need to be initialized. So first we initialize weights. We tend to
# initialize weights as random values in the range of -1 and +1. The tighter
# range the better, we want small values in NN. If we have weights higher than
# 1 your data gets bigger and bigger. We could also normalize and scale our
# input dataset X. We could start with -0.1 and 0.1 for weights is a good start.

# For biases we tend to initialize those as just 0. But 0 also means that
# the output risk to be 0 and if next layer multiplies the 0 with something it
# is still zero, risk that the network is dead, all 0 output. If we have such
# problem we could start biases of with a non 0 number.

# Define the two hidden layers.
class Layer_Dense: # create class
    def __init__(self,n_inputs, n_neurons):
        self.weights = np.random.randn(n_inputs, n_neurons) # we need the shape, size of input, and how many neurons we have
    def forward(self):
        pass

# Lets say we have 4 inputs and 3 neurons, try to print
#print(np.random.randn(4, 3)) # this prints out weights.
# some of these values are bigger than 1.
#print(0.1*np.random.randn(4, 3)) # add 0.1, it will look more like we want <1

# repeat above but add 0.1
class Layer_Dense: # create class object
    def __init__(self,n_inputs, n_neurons):
        # these are our weights
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons) # this is our shape
        # these are our biases
        self.biases = np.zeros((1, n_neurons)) # shape is 1 by n neurons we have
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

# two things to point out before we move to forward method.
# 1. input to np.zeros to pass the shape, the first parameter is the shape,
# so we must pass our shape as a parameter, will be a tuple of the shape
# np.random.rand() the parameters ARE the shapes
# Both 1 and 2 functions returns a matrix of the shape.
# 2. Shape of weights, we shape weights by saying this matrix is n_inputs by
# the n_neurons. It is opposite to the X, n_neurons times n_inputs.
# We dont need to do the transpose, we have full control in the initialization.

# forward methos is going to take inputs. If this is the first layer, n_inputs
# will be actual training data or inputs will be self.output from previous layer.
# self.output is going to be np.dot() which will be inputs multiplied by then
# self.weights + self.biases. That is our forward method.

# Now let´s make use of this. First we make our Layers
layer1 = Layer_Dense(4,5)
# specify size of inputs, how many features in each samples, our case 4
# for n_neurons, anything we want. We go for now 5. This is dynamic and why we
# want to make this an object. You can change it, makes more sense as object.

# output from layer1 is going to be input of layer2.
layer2 = Layer_Dense(5,2)
# the input must be the shape of output of layer1 output. I.e 5.
# the output can be any size we want. We will just say 2. Can pick what we want.

# Now when we have these layer objects we can pass data through these objects.
# When we do that we get an self.output of that layer. Start with layer1.
# We do a forward method and pass in X data. We will then have output value.
layer1.forward(X)
#print(layer1.output) # check output
# We have three samples, we get three outputs of this batch.
# Each one has 5 values, that is how many neurons we said this layer has.
# This now become input to layer to.
layer2.forward(layer1.output)
print(layer2.output) # check output
# We have size three outputs of this batch.
# Each has 2 values, that is the number of neurons we specified.

# We have passed the inputs through our dense forward layer.
# Next major thing we are missing is our activation function.
