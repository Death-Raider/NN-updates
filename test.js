const {NeuralNetwork,LinearAlgebra,Convolution,MaxPool} = require('./Neural-Network.js')
const mxpool = new MaxPool
let input = [[
    [0,0,1,1,0,0],
    [0,0,1,1,0,0],
    [1,1,1,1,1,1],
    [1,1,1,1,1,1],
    [0,0,1,1,0,0],
    [0,0,1,1,0,0]
]] // shape ->  1x6x6
let fake_grads = [
    [ 0 ], [ 1 ],
    [ 0 ], [ 1 ],
    [ 5 ], [ 1 ],
    [ 0 ], [ 1 ],
    [ 0 ]
]
let output = mxpool.pool(input)//other arguments default to 2,2,and true
console.log(output)
let input_grads = mxpool.layerGrads(fake_grads)
console.log(input_grads);
// [
//     [
//         [ 0, 1, 0 ],
//         [ 1, 1, 1 ],
//         [ 0, 1, 0 ]
//     ]
// ]
