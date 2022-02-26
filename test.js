const {NeuralNetwork,LinearAlgebra,Convolution,MaxPool} = require('./Neural-Network.js')
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
console.log(linearA.scalarMatrixProduct(2,a))
console.log(a)
// [ [ 8, 14, 18, 20 ], [ 20, 18, 14, 8 ] ]
