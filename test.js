const {NeuralNetwork,LinearAlgebra,Convolution,MaxPool} = require('./Neural-Network.js')
const conv = new Convolution
let input = [[
    [0,0,1,1,0,0],
    [0,0,1,1,0,0],
    [1,1,1,1,1,1],
    [1,1,1,1,1,1],
    [0,0,1,1,0,0],
    [0,0,1,1,0,0]
]] // shape ->  1x6x6
let filter = [
    [[
        [0,1,0],
        [0,1,0],
        [0,1,0]
    ]],
    [[
        [0,0,0],
        [1,1,1],
        [0,0,0]
    ]]
] // shape -> 2x1x3x3
let output = conv.convolution(input,filter,false,(x)=>x)
console.log(output)
let fake_grads = [
    [
        [0,1,1,0],
        [0,1,1,0],
        [0,1,1,0],
        [0,1,1,0]
    ],
    [
        [0,0,0,0],
        [1,1,1,1],
        [1,1,1,1],
        [0,0,0,0]
    ]
]
const La = new LinearAlgebra
fake_grads = La.vectorize(fake_grads)
fake_grads = La.reconstructMatrix(fake_grads,{x:4*4,y:2,z:1}).flat(1)
fake_grads = La.transpose(fake_grads)
let next_layer_grads = conv.layerGrads(fake_grads)
console.log(next_layer_grads)
