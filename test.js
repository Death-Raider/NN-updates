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
output = conv.convolution(input,filter,false,(x)=>x)
console.log(output)
// [
//   [
//       [ 1, 3, 3, 1 ],
//       [ 2, 3, 3, 2 ],
//       [ 2, 3, 3, 2 ],
//       [ 1, 3, 3, 1 ]
//   ],
//   [
//       [ 1, 2, 2, 1 ],
//       [ 3, 3, 3, 3 ],
//       [ 3, 3, 3, 3 ],
//       [ 1, 2, 2, 1 ]
//   ]
// ]
