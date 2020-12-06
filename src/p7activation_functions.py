# Activation functions:
# Rectified Linear function, Step function, Sigmoid function
# every neuron will have an activation function.
# usually the output layer will have differenct af than the neuronw in
# the hidden layers

# Step function, it´s input and output y = {1 if x > 0 ,0 if x >= 0}
# if input is > 0, the output will be 1
# if input is <= 0, the output will be 0

# using it as activation funtion in the neural networks.
# after we do inputs * weights + bias it comes into play
# (inputs*weights+biases)*activation function
# when we tweak input or biases we can affect the outputs
# output of the activation function 1, 0 becomes input to next neuron

# Sigmoid function, y = 1 / (1 + e~-x)
# Using simoid is reliable due to the grannularity of the output
# We get a more grannular output than the Step function. Next step we will
# count the Loss and then decrease the Loss using a optimizare. Thus a
# Sigmoid is more usefull so it is more grannular.

# Rectified linear function(ReLU), y = { x if x > 0, 0 if x <= 0}
# Looks like a flat, horizontal curve in a 2d space, go linearly up at elbow 0
# Weights or biases can offest the output. Sigmoid have a problem with
# diminishing gradients. ReLU is simpler and faster than Sigmoid.
# A popular activation function for hidden layers in neural networks.

# What is the point of having activation function?
# Without activation funtion the inputs * weights * biases would be a linear
# function y = x, at singular neuron level. All of the outputs would just
# simply be linear. We could not fit a neural network to e.g a sine wave.
# ReLu is like several linear functions with elbow points fititng approximately
# over a sine wave. It is much better than just a linear activation function.
# ReLu is almost linear, but the clipping at 0 is what makes it as powerfull
# as a Sigmoid activation function.
# By negating the weight we flip the slope of the line, by chaning the bias
# we offset the line, and see at what point the input de-activates (- slope)
# at a single neoron point of view.
# neuron 1: input -1.00, bias 0.50, neuron 2: input 1.0, bias 1.0 the total
# output is offset vertically. If we change n2 input to -2.0 we have both a
# upper and lower bound, first flat att 0, linear upp, flat again at y 1.
