# three neurons
# four inputs each
inputs = [1, 2, 3, 2.5]

# three unique weight sets with four values
weights1 = [0.2, 0.8, -0.5, 1.0]
weights2 = [0.5, -0.91, 0.26, -0.5]
weights3 = [-0.26, -0.27, 0.17, 0.87]

# three unique or separate biases
bias1 = 2
bias2 = 3
bias3 = 0.5

# we are modelling three neurons
output = [inputs[0]*weights1[0]+inputs[1]*weights1[1]+inputs[2]*weights1[2]+inputs[3]*weights1[3]+bias1,
         inputs[0]*weights2[0]+inputs[1]*weights2[1]+inputs[2]*weights2[2]+inputs[3]*weights2[3]+bias2,
         inputs[0]*weights3[0]+inputs[1]*weights3[1]+inputs[2]*weights3[2]+inputs[3]*weights3[3]+bias3]
# each neuron will have its on separate weight set and bias and output in the list
print(output)

# inputs are outputs from previous layer of actual input feature data, cant change inputs directly
# by changing weights and biases you can change the inputs indirectly 
# that is the challenge with DL to tweak weights to affect the output
