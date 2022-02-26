# Neural Network
Installing
-----------
```
npm i @death_raider/neural-network
```
About
-----
This is an easy to use Neural Network package with SGD using backpropagation as a gradient computing technique.

Creating the model
------------------
```js
const NeuralNetwork = require('@death_raider/neural-network').NeuralNetwork
//creates ANN with 2 input nodes, 1 hidden layers with 2 hidden nodes and 1 output node
let network = new NeuralNetwork({
  input_nodes : 2,
  layer_count : [2],
  output_nodes :1,
  weight_bias_initilization_range : [-1,1]
});
```
Parameters like the activations for hidden layer and output layers are set as leaky relu and sigmoid respectively but can changed
```js
//format for activation function = [ function ,  derivative of function ]
network.Activation.hidden = [(x)=>1/(1+Math.exp(-x)),(x)=>x*(1-x)] //sets activation for hidden layers as sigmoid function
```
Training, Testing and Using
---------------------------
For this example we'll be testing it on the XOR function.

There are 2 ways we can go about training:

1) Inbuilt Function
```js
function xor(){
  let inp = [Math.floor(Math.random()*2),Math.floor(Math.random()*2)]; //random inputs 0 or 1 per cell
  let out = (inp.reduce((a,b)=>a+b)%2 == 0)?[0]:[1]; //if even number of 1's in input then 0 else 1 as output
  return [inp,out]; //train or validation functions should have [input,output] format
}
network.train({
  TotalTrain : 1e+6, //total data for training (not epochs)
  batch_train : 1, //batch size for training
  trainFunc : xor, //training function to get data
  TotalVal : 1000, //total data for validation (not epochs)
  batch_val : 1, //batch size for validation
  validationFunc : xor, //validation function to get data
  learning_rate : 0.1, //learning rate (default = 0.0000001)
  momentum : 0.9 // momentum for SGD
});
```
The `trainFunc` and `validationFunc` recieve an input of the batch iteration and the current epoch which can be used in the functions.

_`NOTE: The validationFunc is called AFTER the training is done`_

Now to see the avg. test loss:
```js
console.log("Average Validation Loss ->",network.Loss.Validation_Loss.reduce((a,b)=>a+b)/network.Loss.Validation_Loss.length);
// Result after running it a few times
// Average Validation Loss -> 0.00004760326022482792
// Average Validation Loss -> 0.000024864418333478723
// Average Validation Loss -> 0.000026908106414283446
```
2) Iterative
```js
for(let i = 0; i < 10000; i++){
  let [inputs,outputs] = xor()
  let dnn = network.trainIteration({
    input : inputs,
    desired : outputs,
    learning_rate : 0.5
  })
  console.log(dnn.Cost,dnn.layers); //optional to view the loss and the hidden layers
}
// output after 10k iterations
// 0.00022788194782669534 [
//   [ 1, 1 ],
//   [ 0.6856085043616054, -0.6833685003507397 ],
//   [ 0.021348627488749498 ]
// ]
```
This iterative method can be used for visulizations, dynamic learning rate, etc...

To use the network:
```js
// network.use(inputs)  --> returns the hidden node values as well
let output = [ //truth table for xor gate
  network.use([0,0]),
  network.use([0,1]),
  network.use([1,0]),
  network.use([1,1])
]
```

To get the gradients w.r.t the inputs (Questionable correct me if wrong values)
```js
console.log( network.getInputGradients() );
```

Saving and Loading Models
-------------------------
This package allows to save the hyperparameters(weights and bias) in a file(s) and then unpack them, allowing us to use pretrained models.
Saving the model couldnt be further from simplicity:
```js
network.save(path)
```
Loading the model requires a bit more work as it is asynchronous:
```js
const NeuralNetwork = require('./Neural Network/Neural-Network.js')
let network = new NeuralNetwork({
  input_nodes : 2,
  layer_count : [2],
  output_nodes :1,
  weight_bias_initilization_range : [-1,1]
});
(async () =>{
  await network.load(path) //make sure network is of correct structure
  let output = [  
    network.use([0,0]),  // --> returns the hidden node values as well
    network.use([0,1]),  // --> returns the hidden node values as well
    network.use([1,0]),  // --> returns the hidden node values as well
    network.use([1,1])   // --> returns the hidden node values as well
  ]
})()
```

# Linear Algebra
This class is not the most optimized as it can be, but the implementation of certain functions are based on traditional methods to solving them. Those functions will be marked with the * symbol.

Base function
--------------
The base function (basefunc) is a recursive function that takes in 3 parameters a, b, and Opt where a is an array and b is an object and opt is a function. The basefunc goes over all elements of a and also b if b is an array and then passes those elements to the opt function defined by the user. opt will take in 2 parameters and the return can be any object.
```js
linearA = new LinearAlgebra
let a = [
    [1,2,3,4],
    [5,6,7,8]
]
let b = [
    [8,7,6,5],
    [4,3,2,1]
]
function foo(p,q){
    return p*q
}
console.log(linearA.basefunc(a,b,foo))
// [ [ 8, 14, 18, 20 ], [ 20, 18, 14, 8 ] ]
```
Matrix Manipulation
--------------------
1)Transpose(.transpose(matrix))__
It gives the transpose of the matrix (only depth 2).__
2)Scalar Matrix Product(.scalarMatrixProduct(scalar,matrix))__
It gives a matrix which has been multiplied a scalar. Matrix can be of any depth.__
3)Scalar Vector Product(.scalarVectorProduct(scalar,vector))__
It gives a vector(array) which has been multipied by a scalar.__
4)Vector Dot Product(.vectorDotProduct(vec1,vec2))__
It gives the dot product for vectors.__
5)Matrix vector product(.MatrixvectorProduct(matrix,vector))__
It gives the product of a matrix and a vector.__
6)Matrix Product(.matrixProduct(matrix1,matrix2))__
It gives the product between 2 matrices.__
7)Kronecker Product(.kroneckerProduct(matrix1,matrix2))__
It gives the kronecker product of 2 matrices.__
8)Flip(.flip(matrix))__
It flips the matrix by 180 degrees.__
9)Minor*(.minor(matrix,i,j))__
It calculates the minor of a matrix given the index of an element.__
10)Determinant*(.determinant(matrix))__
It calculates the determinant of a matrix using minors.__
11)Invert Matrix*(.invertMatrix(matrix))__
It inverts the matrix using the cofactors.__
12)Vectorize(.vectorize(matrix))__
Vectorizes the matrix by stacking the columns.__
13)im2row & im2col*(.im2row(matrix,[shape_x,shape_y]) / .im2col(matrix,[shape_x,shape_y]))__
Gives the im2row and im2col expansion using a recursive method.__
14)Reconstruct Matrix(.reconstructMatrix(array,{x:x,y:y,z:z}))__
It gives the matrix of the specificed dimension from a flat array.__
15)Normalize(.normalize(matrix,lower_limit,upper_limit))__
It gives the normalized version of the matrix between the specified limits.__
16)Weighted Sum(.weightedSum(weight,matrix1,matrix2,matrix3,...))__
Takes the element from Matrix1 and adds to the element of Matrix2 * weight and then the result is added to the element of Matrix3 * weight and repeated for all given matrices.


Future Updates
--------------
1) Convolution and other image processing functions    ✔️done
2) Convolutional Neural Network (CNN)    ✔️ done
3) Visulization of Neural Network     ❌ pending (next)
4) Recurrent Neural Network (RNN)     ❌ pending
5) Long Short Term Memory (LSTM)    ❌ pending
6) Proper documentation    ❌ pending
