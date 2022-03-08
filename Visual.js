const {NeuralNetwork,LinearAlgebra,Convolution,MaxPool} = require('./Neural-Network.js')
const mnist = require('mnist')
const fs = require('fs')
const cliProgress = require('cli-progress');
// 4:35pm -- 9:00pm
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
let La = new LinearAlgebra;
let conv = new Convolution;
let conv2 = new Convolution;
let mxPool1 = new MaxPool;

// let convData = JSON.parse(fs.readFileSync("Conv/Filter.txt"))
// let conv2Data = JSON.parse(fs.readFileSync("Conv2/Filter.txt"))
//
// conv.x_shape = convData.X_shape
// conv.f_shape = convData.F_shape
// conv.y_shape = convData.Y_shape
// conv.F = convData.Filter
//
// conv2.x_shape = conv2Data.X_shape
// conv2.f_shape = conv2Data.F_shape
// conv2.y_shape = conv2Data.Y_shape
// conv2.F = conv2Data.Filter

let set = mnist.set(5000, 2000)
let Data = []

let P1 = createParameters(8*8*10,[],10,-1,1)
let P2 = createParameters(8*8*10,[],10,-1,1)

let F1_new1 = []
let F1_new2 = []
let f = []
let filter_count_1 = 4
for(let i = 0; i<filter_count_1; i++){
    F1_new1.push(createMatrix(1,5,5,()=>Math.random()))
    F1_new1.push(createMatrix(1,5,5,()=>Math.random()))
    f.push(createMatrix(1,5,5,()=>Math.random()))
}
let F2_new1 = []
let F2_new2 = []
let f2 = []
let filter_count_2 = 10
for(let i= 0; i<filter_count_2; i++){
    F2_new1.push(createMatrix(filter_count_1,5,5,()=>Math.random()))
    F2_new1.push(createMatrix(filter_count_1,5,5,()=>Math.random()))
    f2.push(createMatrix(filter_count_1,5,5,()=>Math.random()))
}

//initilizing network
let network = new NeuralNetwork({
  input_nodes : 8*8*filter_count_2,
  layer_count : [],
  output_nodes :10,
  weight_bias_initilization_range : [-1,1]
});
network.Activation.hidden = [(x)=>1/(1+Math.exp(-x)),(x)=>x*(1-x)]
//Required parameters
function forward_pass(input,desired,filter1,filter2,weight,bias){
    let y1 = conv.convolution(input,filter1,true)
    if(y1.flat(Infinity).filter(e=>e===0).length != y1.flat(Infinity).length)
        y1 = La.normalize(y1,-1,1)
    //Max Pool
    let y2 = mxPool1.pool(y1)
    //conv2
    let y3 = conv2.convolution(y2,filter2,true)
    if(y3.flat(Infinity).filter(e=>e===0).length != y3.flat(Infinity).length)
        y3 = La.normalize(y3,0,1)
    //feed to network and get cost
    network.Weights = weight
    network.Bias = bias
    let out = network.trainIteration({
        input:La.vectorize(y3),
        desired:desired,
    });
    let pred = out.Layers[out.Layers.length-1]

    return [y1,y2,y3,out, pred]
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
function Visual(epoch){
    let inputDrawing = La.normalize(set.training[0].input,-1,1)
    inputDrawing = La.reconstructMatrix(inputDrawing,{x:28,y:28,z:1})
    let drawingOutput = set.training[0].output

    const AddParameterLossVisualization = (P1,P2,P3,alpha,beta) =>{
        let added1 = La.basefunc(P2,P3,(x,y)=>(alpha*x+beta*y))
        return La.basefunc(P1,added1,(x,y)=>(x+y))
    }
    if (Data[epoch] === undefined) Data[epoch] = {cost:[],alpha:[],beta:[]};

    let SavedWeight = JSON.parse(JSON.stringify(network.Weights))
    let SavedBias = JSON.parse(JSON.stringify(network.Bias))
    let SavedFilter1 = JSON.parse(JSON.stringify(conv.F))
    let SavedFilter2 = JSON.parse(JSON.stringify(conv2.F))

    let costVisualSlice = [], resolutionLoss = 10
    for(let a = -resolutionLoss; a < resolutionLoss; a++){
        if(Data[epoch].alpha.length < resolutionLoss*2){
            Data[epoch].beta.push(a/resolutionLoss)
            Data[epoch].alpha.push(a/resolutionLoss)
        }
        for(let b = -resolutionLoss; b < resolutionLoss; b++){
            let newParameters = AddParameterLossVisualization([SavedFilter1,SavedFilter2,[SavedWeight,SavedBias]],[F1_new1,F2_new1,P1],[F1_new2,F2_new2,P2],a,b)
            let [x,xx,xxx,result,xxxx] = forward_pass(inputDrawing,drawingOutput,newParameters[0],newParameters[1],newParameters[2][0],newParameters[2][1])
            costVisualSlice.push(result.Cost)
        }
    }
    Data[epoch].cost.push(costVisualSlice.slice())
    fs.writeFileSync('logs\\Info2.json', JSON.stringify(Data));
    conv.F = SavedFilter1
    conv2.F = SavedFilter2
    network.Weights = SavedWeight
    network.Bias = SavedBias
}
function Train(BATCH_SIZE,EPOCH,momentum){
    let acc = {t:0,f:0}
    let BATCH_Stack = {
        Filters:[],
        FullyConnected:[]
    }
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
        for(let step = 0; step < set.training.length; step++){

            let x = La.normalize(set.training[step].input,-1,1)
            x = La.reconstructMatrix(x,{x:28,y:28,z:1})
            let desired = set.training[step].output

            let [y1,y2,y3,out, pred] = forward_pass(x,desired,f,f2,network.Weights,network.Bias)
            cost = out.Cost.toFixed(3)

            if(step%1000 == 0){
                // every 1000 steps reset accuracy and fill logs
                acc = {t:0,f:0}
                // fs.writeFileSync('logs\\Info.json', JSON.stringify(Data));
            }
            if(step%25==0)Visual(epoch)
            if(step%BATCH_SIZE == 0){
                if(step != 0){// every bacth if step is not 0
                    //getting initial changes
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
                    network.update(WeightUpdate,BiasUpdate,0.1);
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
        //Save network
        network.save("Net1")
        conv.saveFilters("Conv")
        conv2.saveFilters("Conv2")
        mxPool1.savePool("Pool")
    }
}

async function Start(){
    // console.log(network.Weights);
    // await network.load("Net1")
    Train(1,1,0.9)
}
Start()
