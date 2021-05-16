# nnfs
Building neural networks from scratch by way of nnfs.io.  
Folder structure and description of key concepts and code snippets.     

#### folders  
* [x] 0.basics  
* [x] 2.coding first neurons  
* [x] 3.adding Layers  
* [x] 4.activation functions  
* [x] 5.calculating network error with loss  
* [x] 6.introducing optimization  
* [x] 7.derivatives  

#### shape ####  
Understanding shape.
Array: l = [1,5,6,2]  
Shape: (4,)  
Type: list is a 1D array in numpy, Vector in math  

Array: lol = [[1,5,6,2],[3,2,1,3]]  
Shape: (2, 4)  
Type: list-of-list is a 2D array in numpy, Matrix in math  

 Array: lolol = [[[1,5,6,2],[3,2,1,3]],[[1,5,6,2],[3,2,1,3]],[[1,5,6,2],[3,2,1,3]]]  
 Shape: (3, 2, 4)  
 Type: list-of-list-of-list is a 3D array in numpy  
 At the first element we have three dimensions, at the second we have two, at the third we have four.  

A tensor can be represented as an array, we will work with tensor in this form.  


#### terminology ####  
Layers of neurons.  
Layer size: 10,8,8,8,2. With 8x3 hidden layers.  
Weights: 224, unique connections/lines between neurons.  
Biases: 36, unique neurons.  
Parameters: 260, unique tunable parameters.  

This gives us complex relations that can in theory be mapped. The question is how to tune such a thing. We can do that by using activation functions and an optimizer.    
