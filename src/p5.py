# Now we go from a single sample of input to a batch of inputs
# Go from single layer of neurons to model two layers of neurons
# We will go to object oriented programming, creating our layer object

# why batch?
# batch allow us to calculate in parallel, this is why we tend to do neural
# network computations on gpu´s rather than cpu´s. A cpu may hae 4-8 cores
# while gpu typically can have hundreds of cores. We do typically Matrix
# multiplication which can be done on gpus.
# alos it helps with generalization to use batches.

# one single sample of four features
inputs = [1, 2, 3, 2.5] # current status, we want to pass a batch of these
# samples instead. It helps generalize instead of showing our machine once
# at the time

# Take a sample batch of 512, ask one neuron to fit to that sample (line)
# We are fitting one sample at the time to a neuron, bouncing around up n down
# one sample at the time draw the fitted line. If we use multiple samples
# it can probably do a better job at fitting the samples.
# batch size: 4, fitted line will still move but less than with size 1.
# batch size: 15, even more stabile.
# batch size 32 at the time is easier than 1 sample, but it can also create
# overfitting. We should not show all samples at once either.
import numpy as np
# batch of inputs
inputs = [[1, 2, 3, 2.5],
          [2.0, 5.0, -1.0, 2.0],
          [-1.5, 2.7, 3.3, -0.8]]
# Do we need to change weights and biases now? No
weights = [[0.2, 0.8, -0.5, 1.0],
           [0.5, -0.91, 0.26, -0.5],
           [-0.26, -0.27, 0.17, 0.87]]

biases = [2, 3, 0.5]
# Both our weights and inputs will be a matrix
# Matrix product. We take first input matrix row vector and we do then
# dot product of every column vector of the second matrix.
# [row1 value1, value2, value3 * [col1 value1, value2, value3] = scalar value1_1
# [row1 value1, value2, value3 * [col2 value1, value2, value3] = scalar value1_2
# [row1 value1, value2, value3 * [col3 value1, value2, value3] = scalar value1_3
# row 2, same thing column wise, etc untill we have an entire output matrix
# with the dot products of the row and column vectors

# Now inputs and weights are currently the same shape, unlike before in readme

# We want to swap rows and columns of weights matrix, since rows in input are 4
# while cols in weights are 3. We get a shape error now when we try dot product
# T swaps row and makes it a col

#output = np.dot(weights, inputs) + Biasesprint(output) # gives shape error

# To transpose is that we need to convert the weights to arrays
output = np.dot(inputs, np.array(weights).T) + biases
print(output) # a batch of outputs in a 3x3 matrix
# transposed weights
# [0.2, 0.5, -0.26]
# [0.8, -0.91, -0.27]
# [-0.5, 0.26, 0.17]
# [1.0, -0.5, 0.87]
# Then we add the biases vector to each of the rows of the matrix product outputs
