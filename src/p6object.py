# Create an object, keeping the input as our sample data delete the rest

import numpy as np

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


# Define the two hidden layers.
class Layer_Dense: # create class
    def __init__(self):
        pass # we say pass for Now
    def forward(self):
        pass

# For biases we tend to initialize those as just 0. But 0 also means that
# the output risk to be 0 and if next layer multiplies the 0 with something it
# is still zero, risk that the network is dead, all 0 output. If we have such
# problem we could start biases of with a non 0 number.
