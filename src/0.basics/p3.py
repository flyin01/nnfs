# simplfication of code
inputs = [1, 2, 3, 2.5]

# list of three weight lists
weights = [[0.2, 0.8, -0.5, 1.0],
           [0.5, -0.91, 0.26, -0.5],
           [-0.26, -0.27, 0.17, 0.87]]

# three unique or separate biases
biases = [2, 3, 0.5]

# dynamic way of doing inputs * weight + bias for a layer using loops
layer_outputs = [] # Output of current layers
for neuron_weights, neuron_bias in zip(weights, biases):
    neuron_output = 0  # Output of given neurons
    for n_input, weight in zip(inputs, neuron_weights):
        neuron_output += n_input*weight
    neuron_output += neuron_bias
    layer_outputs.append(neuron_output)

print(layer_outputs)

# weights and biases are the nobs of tunable parameters
# weight is a multiple, it can stay negative of a starting values
# bias is an offset, we can go from negative to positive for example
# weights and biases are two different tools that help in different ways
# this will be clearer when we get into activation functions
