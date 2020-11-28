# build a neuron in a feed-forward multi layered models

# three neuron´s outputs become the input of another neurons
# making up some numbers as examples
inputs = [1.2, 5.1, 2.1]
weights = [3.1, 2.2, 8.5]
bias = 3 # every unique neuron has a unique bias

# simple output of the neuron by adding inputs * weights + bias
output = inputs[0]*weights[0] + inputs[1]*weights[1] + inputs[2]*weights[2] + bias
print(output)
