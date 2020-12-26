# ReLu Activation

import numpy as np

np.random.seed(0)

# Batch of inputs
X = [[1, 2, 3, 2.5],
     [2.0, 5.0, -1.0, 2.0],
     [-1.5, 2.7, 3.3, -0.8]]

# Write simple Rectified Linear functions
inputs = [0, 2, -1, 3.3, -2.7, 1.1, 2.2, -100]
output = [] #empty lists

# alt 1
for i in inputs:
    if i > 0:
        output.append(i)
    elif i <= 0:
        output.append(0)

print(output)
# all the output is clipped from 0 and below to be 0

# alt 2
output = []
for i in inputs:
    output.append(max(0, i))

print(output) # this gives exact same output as alt 1

# Next we write the rectified linear object
