// const { createCanvas } = require('canvas')
const {NeuralNetwork, LinearAlgebra,  Convolution, MaxPool} = require("./Neural-Network.js")
const fs = require('fs')
const cliProgress = require('cli-progress');


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

network.Activation.hidden = [(x)=>1/(1+Math.exp(-x)),(x)=>x*(1-x)]

let P1 = createParameters(8*8*10,[100],10,-1,1)
let P2 = createParameters(8*8*10,[100],10,-1,1)

let F1_new1 = []
let F1_new2 = []
let f1 = []
let filter_count_1 = 4
for(let i = 0; i<filter_count_1; i++){
    F1_new1.push(createMatrix(1,5,5,()=>Math.random()))
    F1_new1.push(createMatrix(1,5,5,()=>Math.random()))
    f1.push(createMatrix(1,5,5,()=>Math.random()))
}
let filter_count_2 = 10
let F2_new1 = []
let F2_new2 = []
let f2 = []
for(let i= 0; i<filter_count_2; i++){
    F2_new1.push(createMatrix(filter_count_1,5,5,()=>Math.random()))
    F2_new1.push(createMatrix(filter_count_1,5,5,()=>Math.random()))
    f2.push(createMatrix(filter_count_1,5,5,()=>Math.random()))
}

let set = mnist.set(8000,0)
let inp = La.normalize(set.training[0].input,-1,1)
inp = La.reconstructMatrix(inp,{x:28,y:28,z:1})
let out = set.training[0].output

const param = (P1,P2,P3,alpha,beta) =>{
    let added1 = La.basefunc(P2,P3,(x,y)=>(alpha*x+beta*y))
    return La.basefunc(P1,added1,(x,y)=>(x+y))
}

Train()

function forward_pass(input,desired,All_Parameters){

    network.Weights = All_Parameters[2][0]
    network.Bias = All_Parameters[2][1]

    let y1 = conv.convolution(input,All_Parameters[0])
    // console.log(Math.max(...input.flat(Infinity)));
    console.log(conv.F);

    if(y1.flat(Infinity).filter(e=>e===0).length != y1.flat(Infinity).length)
        y1 = La.normalize(y1,-1,1)
    //Max Pool
    let y2 = mxPool1.pool(y1)
    //conv2
    let y3 = conv2.convolution(y2,All_Parameters[1])
    if(y3.flat(Infinity).filter(e=>e===0).length != y3.flat(Infinity).length)
        y3 = La.normalize(y3,0,1)
    //feed to network and get cost
    let out = network.trainIteration({
        input:La.vectorize(y3),
        desired:desired,
    });
    let pred = out.Layers[out.Layers.length-1]
    return [y1,y2,y3,out,pred]
}
function backword_pass(){
    //getting gradients from network and reshape into proper format
    let grads_y3 = network.getInputGradients()
    grads_y3 = La.reconstructMatrix(grads_y3,{x:8*8,y:filter_count_2,z:1}).flat(1)
    grads_y3 = La.transpose(grads_y3)
    //sending grads for conv
    let grads_y2 = conv2.layerGrads(grads_y3)
    grads_y2 = La.vectorize(grads_y2)
    grads_y2 = La.reconstructMatrix(grads_y2,{x:12*12,y:filter_count_1,z:1}).flat(1)
    grads_y2 = La.transpose(grads_y2)
    //sending grads to pool
    let grads_y1 = mxPool1.layerGrads(grads_y2)
    //sendin grads to conv
    let grads_x = conv.layerGrads(grads_y1)
    //no point in sending grads to input layer but still doing it
    grads_x = La.vectorize(grads_x)
    grads_x = La.reconstructMatrix(grads_x,{x:28*28,y:1,z:1}).flat(1)
    grads_x = La.transpose(grads_x)

    return {grads_y3,grads_y2,grads_y1,grads_x}
}
function Train(){
    let momentum = 0.9
    let BATCH_SIZE = 1
    let EPOCH = 1
    let BATCH_Stack = {
        Filters:[],
        FullyConnected:[]
    }
    let Total = 10
    let Data = [{alpha:[],cost:[],beta:[]}]
    let acc = {t:0,f:0}
    for(let epoch = 0; epoch < EPOCH; epoch++){
        //setting up the the progress bar
        let bar1 = new cliProgress.SingleBar({
            format: 'Epoch:{epoch} [{bar}] {percentage}% | ETA: {eta}s | {value}/{total} | Acc: {acc}% | Loss: {loss}'
        }, cliProgress.Presets.shades_classic);
        bar1.start(set.training.length, 0,{
            epoch:epoch,
            acc:acc.t*BATCH_SIZE/10,
            loss:0
        })
        acc = {t:0,f:0} // reset accuracy every epoch
        let cost = 0
        for(let i = -Total; i < Total; i++){
            Data[epoch].alpha.push((i/Total)*5)
            Data[epoch].beta.push((i/Total)*5)
        }
        for(let step = 0; step < set.training.length; step++){

            let x = La.normalize(set.training[step].input,-1,1)
            x = La.reconstructMatrix(x,{x:28,y:28,z:1})
            let desired = set.training[step].output
            let params = [network.copyRadar2D(f1),network.copyRadar2D(f2),[network.copyRadar3D(network.Weights),network.copyRadar2D(network.Bias)]]
            let [y1,y2,y3,out, pred] = forward_pass(x,desired,params)
            cost = out.Cost.toFixed(3)

            if(step%1000 == 0){
                // every 1000 steps reset accuracy and fill logs
                acc = {t:0,f:0}
            }
            if(step%1000 == 0){
                network.save("Net1")
                conv.saveFilters("Conv")
                conv2.saveFilters("Conv2")
                mxPool1.savePool("Pool")
            }

            if(step%BATCH_SIZE == 0){
                if(step != 0){// every bacth if step is not 0
                    //getting initial changes
                    console.log(BATCH_Stack.Filters[0].grads_y1);
                    let WeightUpdate =  BATCH_Stack.FullyConnected[0].WeightUpdate
                    let BiasUpdate =  BATCH_Stack.FullyConnected[0].BiasUpdate
                    let F1 = BATCH_Stack.Filters[0].grads_y1
                    let F2 = BATCH_Stack.Filters[0].grads_y3
                    for(let i = 1; i < BATCH_SIZE; i++){// adding the other weights of the batch together
                        WeightUpdate = La.basefunc(WeightUpdate,BATCH_Stack.FullyConnected[i].WeightUpdate,(x,y)=>x+y)
                        BiasUpdate = La.basefunc(BiasUpdate,BATCH_Stack.FullyConnected[i].BiasUpdate,(x,y)=>x+y)
                        F1 = La.basefunc(F1,BATCH_Stack.Filters[i].grads_y1,(x,y)=>x+y)
                        F2 = La.basefunc(F2,BATCH_Stack.Filters[i].grads_y3,(x,y)=>x+y)
                    }
                    //Applying momentum
                    WeightUpdate = La.basefunc(BATCH_Stack.FullyConnected[BATCH_SIZE-1].WeightUpdate,WeightUpdate,(a,b)=>(a*momentum + b*(1-momentum)))
                    BiasUpdate = La.basefunc(BATCH_Stack.FullyConnected[BATCH_SIZE-1].BiasUpdate,BiasUpdate,(a,b)=>(a*momentum + b*(1-momentum)))
                    F1 = La.basefunc(BATCH_Stack.Filters[BATCH_SIZE-1].grads_y1,F1,(a,b)=>(a*momentum + b*(1-momentum)))
                    F2 = La.basefunc(BATCH_Stack.Filters[BATCH_SIZE-1].grads_y3,F2,(a,b)=>(a*momentum + b*(1-momentum)))
                    //updating
                    network.update(WeightUpdate,BiasUpdate,0.01);
                    conv2.filterGrads(F2,1e-4)
                    conv2.F = La.normalize(conv2.F,-1,1)
                    
                    conv.filterGrads(F1,1e-3)
                    conv.F = La.normalize(conv.F,-1,1)
                    //update Accuracy
                    if(pred.indexOf(Math.max(...pred))==desired.indexOf(Math.max(...desired))){
                        acc.t += 1
                    }else{
                        acc.f += 1
                    }
                    //redefining data for new epoch
                    if(step%100 == 0){
                        // network.Weights[0][0][0] = 99
                        // if (Data[epoch] === undefined)Data[epoch] = {cost:[],alpha:[],beta:[]};
                        // //updating Data
                        // Data[epoch].cost.push([])
                        // for(let a of Data[epoch].alpha){
                        //     for(let b of Data[epoch].beta){
                        //         [a,b] = [parseFloat(a),parseFloat(b)]
                        //         let All_Parameters = param([conv.F,conv2.F,[network.Weights,network.Bias]],[F1_new1,F2_new1,P1],[F1_new2,F2_new2,P2],a,b)
                        //         let [_1,_2,_3,result,_4] = forward_pass(inp,desired,All_Parameters)
                        //         Data[epoch].cost[Data[epoch].cost.length-1].push(result.Cost)
                        //     }
                        // }
                        // fs.writeFileSync('logs\\Info2.json', JSON.stringify(Data));
                    }

                    conv.F = params[0]
                    conv2.F = params[1]
                    network.Weights = params[2][0]
                    network.Bias = params[2][1]
                }
                //Reinitilizing Batch
                BATCH_Stack.Filters = []
                BATCH_Stack.FullyConnected = []
                //move pregress bar by batch size and show the following
                bar1.increment(BATCH_SIZE,{epoch:epoch,loss:out.Cost.toFixed(3),acc:acc.t*BATCH_SIZE/10})
            }
            //update batch
            BATCH_Stack.Filters.push(backword_pass())
            BATCH_Stack.FullyConnected.push({WeightUpdate:network.WeightUpdates,BiasUpdate:network.BiasUpdates})
        }
        //stop bar at end of epoch
        bar1.stop()

        // print out details
        console.log(
            "cost:",cost,
            "epoch",epoch,
            "acc:",acc.t*BATCH_SIZE/10,"%",acc.f*BATCH_SIZE/10,"%"
        );
        //Write in logs
        fs.writeFileSync('logs\\Info2.json', JSON.stringify(Data));

        //Save network
        network.save("Net1")
        conv.saveFilters("Conv")
        conv2.saveFilters("Conv2")
        mxPool1.savePool("Pool")
    }
}





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
