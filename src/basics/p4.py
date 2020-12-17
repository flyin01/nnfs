# how numpy version of dot product works

# We are trying to do inputs * weight * bias
# We want to multiply this vector * matrix,  how do we do that
# We use the dot product (matrix product), in terms of numpy.

# dot product examples. multiply vectors element wise and add that together
a = [1, 2, 3]
b = [2, 3, 4]

# the dot product results in a scalar single value
dot_product = a[0]*b[0] + a[1]*b[1] + a[2]*b[2]
# [1,2,3] * [2,3,4] = 1*2 + 2*3 + 3*4 = 20
print(dot_product)

# dot product of simplified version of inputs, weights, bias for one neuron
import numpy as np
inputs = [1, 2, 3, 2.5]
weights = [0.2, 0.8, -0.5, 1.0]
bias = 2

output = np.dot(weights, inputs) + bias
# = 0.2*1.0 + 0.8*2.0 + -0.5*3.0 + 1.0*2.5 = 2.0 + 2.0 = 4.8
print(output)

# next, do dot product of layer of neuron_bias
inputs = [1, 2, 3, 2.5]           # vector
weights = [[0.2, 0.8, -0.5, 1.0], # matrix of vectors
           [0.5, -0.91, 0.26, -0.5],
           [-0.26, -0.27, 0.17, 0.87]]

bias = [2, 3, 0.5]

output = np.dot(weights, inputs) + bias
print(output)
# [4.8, 1.21, 2.385]

# the first element you return in np.dot(weights, ...) is how the return is
# going to be indexed. We are modelling three neurons, three sets of weights is
# how we know this is three neurons (before we do the bias). We want output to
# be the values from the neurons, that is why we pass weights as first argument.
# If we pass it the other way around, we get some weird shape error.

# Comment in and out below row to see error!
#output = np.dot(inputs, weights) + bias
print(output)
# ValueError: shapes (4,) and (3,4) not aligned: 4 (dim 0) != 3 (dim 0)

# the dot product of the correct order is going to be
# weights first vector and inputs vector, weights second vector and inputs etc..
# we iterate through the matrix of vectors and take the dot product three times
# np.dot(weights, inputs) = [np.dot(weight[0], inputs),
# np.dot(weights[1], inputs), np.dot(weights[2],inputs)]
# = [2.8, -1.79, 1.885] then we + biases to get the output which is our final values

# Now we know how it works for one layer, soon we will do this on a batch.

# Until now our input data has been a 1D array
# Input data:
sample = [1,5,6,2]    # Shape: (4,)  Type: 1D array, vector

# Input data batch:
batch = [[1,5,6,2],  # Shape 4  Type: 2D array, Matrix
         [3,2,1,3],
         [5,2,1,2],
         [6,4,8,4]]

# output = weight * input + bias , is similar to the straight line equation
# if we modify the weight, it impacts the weight like the slope of a line
# if we adjust bias, it offsets the line like when changing the y-intercept

# benefits and differences of weights vs biases comes even more apparent once
# we get to activation functions:
# output = ReLU(weight * input + bias)
# Activation point can be like where a line is bent like elbow, horizonal line
# goes of upwards at e.g -2 x.
