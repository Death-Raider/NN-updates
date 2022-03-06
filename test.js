// const { createCanvas } = require('canvas')
const {NeuralNetwork, LinearAlgebra,  Convolution, MaxPool} = require("./Neural-Network.js")
const fs = require('fs')

function make_circle(size){
    let radius = Math.floor(Math.random()*size/2)
    let coords = {
        x: Math.floor(Math.random()*(size-2*radius)+radius),
        y: Math.floor(Math.random()*(size-2*radius)+radius)
    }
    let inCircle = (point,radius,center) => ( radius >= Math.hypot((point.x-center.x),(point.y-center.y)) )
    let Matrix = []
    for(let i = 0; i < size; i++){
        Matrix[i] = []
        for(let j = 0; j < size; j++){
            Matrix[i][j] = 0
            if(inCircle({x:j,y:i},radius,coords)) Matrix[i][j] = 1
        }
    }
    return Matrix
}
function make_triangle(size){
    let top = Math.floor(Math.random()*(size-2)+1)
    let left = Math.floor(Math.random()*(size-2)+1)
    let Matrix = Array(size).fill(0).map(e=>Array(size).fill(0).map(e=>0))
    for(let i = top, k = 0; i < size; i++,k++){
        for(let j = left-k; 2*k+1+left-k < size && j < 2*k+1+left-k && j > 0; j++){
            Matrix[i][j] = 1
        }
    }
    return Matrix
}
function draw_matrix_2d(matrix){
    const { createCanvas } = require('canvas')
    const canvas = createCanvas(matrix.length, matrix[0].length)
    const ctx = canvas.getContext('2d')
    function save(canvas){
      const buffer = canvas.toBuffer('image/png')
      fs.writeFileSync('image.png', buffer)
    }
    for(let i = 0; i < matrix.length; i++){
        for(let j = 0; j < matrix[i].length; j++){
            ctx.fillStyle = (matrix[i][j] == 0)?"black":"white"
            ctx.fillRect(j,i,1,1)
        }
    }
    save(canvas)
}
function createParameters(input,LayerCount,output,a,b){
  let MatrixW=[], MatrixB=[];
  for(let j = 0; j < LayerCount.length + 1; j++){
    MatrixW[j]=[]; MatrixB[j]=[];
    for(let i = 0; i < ((j == LayerCount.length)?output:LayerCount[j]);i++){
      MatrixW[j][i]=[];
      MatrixB[j][i]=Math.random()*(b-a)+a;
      for(let k = 0; k < ((j == 0)?input:LayerCount[j-1]); k++) MatrixW[j][i][k]=Math.random()*(b-a)+a;
    }
  }
  return [MatrixW,MatrixB];
}
function make_figures(){
    let r = Math.floor(Math.random()*2)
    let input = r?make_circle(15):make_triangle(15)
    // draw_matrix_2d(input)
    input = input.flat()
    let output = r?[1,0]:[0,1]
    return [input,output]
}
function createMatrix(z,y,x,value){
    let M = []
    for(let i = 0; i < z; i++){
        M[i] = []
        for(let j = 0; j < y; j++){
            M[i][j] = []
            for(let k = 0; k < x; k++){
                M[i][j][k] = value()
            }
        }
    }
    return M
}

const La = new LinearAlgebra
let network = new NeuralNetwork({
    input_nodes: 8*8*10,
    layer_count: [100],
    output_nodes: [10]
})

const mnist = require('mnist')
const conv = new Convolution
const conv2 = new Convolution
const mxPool1 = new MaxPool

let trained = JSON.parse(fs.readFileSync("logs/Info.json"))[0]

let F1 = trained.F1
let F2 = trained.F2
let FC = [trained.W,trained.B]
conv.x_shape = [28,28,1]
conv.y_shape = [24,24,4]
conv.f_shape = [5,5,1,4]
conv2.x_shape = [12,12,4]
conv2.y_shape = [8,8,10]
conv2.f_shape = [5,5,4,10]


network.Activation.hidden = [(x)=>1/(1+Math.exp(-x)),(x)=>x*(1-x)]

let P1 = createParameters(8*8*10,[100],10,-1,1)
let P2 = createParameters(8*8*10,[100],10,-1,1)
let F1_new1 = []
let F1_new2 = []
let filter_count_1 = 4
for(let i = 0; i<filter_count_1; i++){
    F1_new1.push(createMatrix(1,5,5,()=>Math.random()))
    F1_new2.push(createMatrix(1,5,5,()=>Math.random()))
}
let filter_count_2 = 10
let F2_new1 = []
let F2_new2 = []
for(let i= 0; i<filter_count_2; i++){
    F2_new1.push(createMatrix(filter_count_1,5,5,()=>Math.random()))
    F2_new2.push(createMatrix(filter_count_1,5,5,()=>Math.random()))
}

let set = mnist.set(1,0)
let inp = La.normalize(set.training[0].input,-1,1)
inp = La.reconstructMatrix(inp,{x:28,y:28,z:1})
let out = set.training[0].output

const param = (P1,P2,P3,alpha,beta) =>{
    let added = La.basefunc(P2,P3,(x,y)=>(alpha*x+beta*y))
    return La.basefunc(P1,added,(x,y)=>(x+y))
}

function forward_pass(input,desired,All_Parameters){

    conv.F = All_Parameters[0]
    conv2.F = All_Parameters[1]
    network.Weights = All_Parameters[2][0]
    network.Bias = All_Parameters[2][1]

    let y1 = conv.convolution(input)
    if(y1.flat(Infinity).filter(e=>e===0).length != y1.flat(Infinity).length)
        y1 = La.normalize(y1,-1,1)
    //Max Pool
    let y2 = mxPool1.pool(y1)
    //conv2
    let y3 = conv2.convolution(y2)
    if(y3.flat(Infinity).filter(e=>e===0).length != y3.flat(Infinity).length)
        y3 = La.normalize(y3,0,1)
    //feed to network and get cost

    let out = network.trainIteration({
        input:La.vectorize(y3),
        desired:desired,
    });

    return out
}

let Total = 200
let Data = {alpha:[],cost:[],beta:[]}
for(let i = 0; i < Total; i++){
    let alpha = 20*i/Total - 20/2
    let result
    Data.beta.push([])
    Data.cost.push([])
    for(let j = 0; j < Total; j++){
        let beta = 20*j/Total - 20/2

        let All_Parameters = param([F1,F2,FC],[F1_new1,F2_new1,P1],[F1_new2,F2_new2,P2],alpha,beta)

        result = forward_pass(inp,out,All_Parameters)
        Data.cost[i].push(result.Cost)
        Data.beta[i].push(beta)
    }
    console.log(i,result.Cost)
    Data.alpha.push(alpha)
}
fs.writeFileSync('logs\\Info2.json', JSON.stringify(Data));


// network.train({
//     TotalTrain: 10000,
//     TotalVal: 1000,
//     trainFunc: make_figures,
//     validationFunc: make_figures,
//     batch_train:1,
//     learning_rate: 0.1,
//     momentum: 0.9
// })
// console.log("Average Validation Loss ->",network.Loss.Validation_Loss.reduce((a,b)=>a+b)/network.Loss.Validation_Loss.length);
// fs.writeFileSync('Trained.json', JSON.stringify({
//     W: network.Weights,
//     B: network.Bias
// }));

// let trained = JSON.parse(fs.readFileSync("Trained.json"))
// let Parameters1 = [trained.W,trained.B]
// let Parameters2 = createParameters(15*15,[10],2,-1,1)
// let Parameters3 = createParameters(15*15,[10],2,-1,1)
// let [inp,out] = make_figures()
// let Data = {alpha:[],cost:[],beta:[]}
// let Total = 200
// for(let i = 0; i < Total; i++){
//     let alpha = 2*i/Total - 2/2
//     Data.beta.push([])
//     Data.cost.push([])
//     let result
//     for(let j = 0; j < Total; j++){
//         let beta = 2*j/Total - 2/2
//         const param = (alpha,beta) =>{
//             let added = La.basefunc(Parameters2,Parameters3,(x,y)=>(alpha*x+beta*y))
//             return La.basefunc(Parameters1,added,(x,y)=>(x+y))
//         }
//         let Parameter = param(alpha,beta)
//         network.Weights = Parameter[0]
//         network.Bias = Parameter[1]
//         result = network.trainIteration({
//             input: inp,
//             desired: out
//         })
//         Data.cost[i].push(result.Cost)
//         Data.beta[i].push(beta)
//     }
//     console.log(i,result.Cost);
//     Data.alpha.push(alpha)
// }
//
// fs.writeFileSync('logs\\Info.json', JSON.stringify(Data));
