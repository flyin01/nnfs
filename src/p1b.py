# build a neuron in a feed-forward multi layered models

# three neuron´s outputs become the input of another neurons
# making up some numbers as examples
inputs = [1, 2, 3, 2.5]
weights = [0.2, 0.8, -0.5, 1.0]
bias = 2 # every unique neuron has a unique bias

# simple output of the neuron by adding inputs * weights + bias
output = inputs[0]*weights[0] + inputs[1]*weights[1]\
 + inputs[2]*weights[2] + inputs[3]*weights[3] + bias
print(output)

# nn will randomly initialize Weights
# tweaking those weights is backpropagation
