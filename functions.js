const add = (a, b, k=1) => a instanceof Array ? a.map((c, i) => add(a[i], b[i])) : a + k*b;
function callFile(open,filePath,sendData = 'no Data',resolve,reject){
  const test = spawn(open,[filePath]);
  let x;
  test.stdin.write(sendData);
  test.stdin.end();
  test.stdout.on('data',(data) =>{x = data.toString('utf8')});
  test.stdout.on('end',_=>{resolve(x)});
  test.stderr.on('error',(err)=>{reject(err)});
}
function createMatrix(z,y,x,val){
  let M = []
  for(let i = 0; i < z; i++){
    M[i] = []
    for(let j = 0; j < y; j++){
      M[i][j] = []
      for(let k = 0; k < x; k++){
        M[i][j][k] = val(k)
      }
    }
  }
  return M;
}
function Flip(matrix){
  let newMatrix = []
  matrix.forEach((e,i,a)=>{newMatrix[a.length-(i+1)] = e.slice().reverse()})
  return newMatrix
}
function FastFlip(matrix){
  const reversed=(a)=>a.slice(0).reverse()
  return reversed(matrix).map(reversed)
}
const norm = (T, a, b, opt) => T instanceof Array ? T.map((c, i) => norm(T[i], a, b, opt)) : ( (b-a)*(T-opt.min)/(opt.max-opt.min) + a);
function Rgb(r,g,b){
  r = r.toString(16);
  g = g.toString(16);
  b = b.toString(16);
  if (r.length == 1)
    r = "0" + r;
  if (g.length == 1)
    g = "0" + g;
  if (b.length == 1)
    b = "0" + b;
  return "#" + r + g + b;
}
function addColor(value){
  let color,g;
  if(value == 0) color = Rgb(255,255,255)
  if(value < 0){
    g = (value < -1)?0:parseInt((value+1)*255)
    color = Rgb(g,g,255);
  }
  if(value > 0){
    g = (value > 1)?255:parseInt(value*255)
    color = Rgb(255,255-g,255-g);
  }
  return color;
}
function fib(n){
  if (n==0) return 0
  if (n==1) return 1
  return fib(n-1)+fib(n-2)
}
const basefunc = (a,b,opt) => a instanceof Array ? a.map((c, i) => basefunc(a[i], Array.isArray(b)?b[i]:b, opt)) : opt(a,b) ;
function* zip(...arrs){
  for(let i = 0; i < arrs[0].length; i++){
    let a = arrs.map(e=>e[i])
    if(a.indexOf(undefined) == -1 ){yield a }else{return undefined;}
  }
}
function makeTriangle(s){
  let m = Array(s).fill(0).map(e=>Array(s).fill(0).map(e=>0));
  let t = [Math.floor(Math.random()*s),Math.floor(Math.random()*s)] // random indexs [i,j] for tip of triangle
  if(t[1] != 0 && t[1] != s-1 && t[0] != s-1){
    m[t[0]][t[1]] = 1
    for(let i = t[0], q=0; i < s; i++, q++){
      let j = t[1] - q
      if(j < 0 || j + 2*q + 1 > s) break
      for(let l = 0; l < 2*q + 1; l++) m[i][j+l] = 1
    }
  }else{ return makeTriangle(s) }
  return m
}
function makeCircle(s){
  let m = Array(s).fill(0).map(e=>Array(s).fill(0).map(e=>0));
  // random indexs [i,j] for center of circle
  let t = [Math.floor(Math.random()*(s/2)+s/4),Math.floor(Math.random()*(s/2)+s/4),Math.floor(Math.random()*(s/2)+s/4)]
  t[2] = Math.min(t[0],s-t[0],t[1],s-t[1],t[2]) // gets the min radius
  if( t[2] == 0|| t[1]+t[2] >= s || t[0]+t[2] >= s ){return makeCircle(s)}
  else{
    for(let i = 0; i < s; i++)
      for(let j = 0; j < s; j++)
        if( Math.hypot(t[0]-i,t[1]-j) <= t[2] ) m[i][j] = 1
    return m
  }
}
function drawMatrix(M,name){
  const {createCanvas} = require('canvas');
  const fs = require("fs")

  const save = (canvas,name)=>{
    const buffer = canvas.toBuffer('image/png')
    fs.writeFileSync(`Images\\${name}.png`, buffer)
    return true
  }
  for(let l = 0; l < M.length; l++){
    let canvas = createCanvas(M[l].length, M[l][0].length)
    let ctx = canvas.getContext('2d')
    for(let k = 0; k < M[l].length; k++){
      for(let p = 0; p < M[l][0].length; p++){
        ctx.fillStyle = addColor(M[l][k][p])
        ctx.fillRect(p,k,1,1)
      }
    }
    save(canvas,`${name}_${l}`)
  }
}

const {NeuralNetwork,LinearAlgebra,Convolution,MaxPool} = require('./Neural-Network.js')
const mnist = require('mnist')
const fs = require('fs')
const cliProgress = require('cli-progress');

let La = new LinearAlgebra;
let conv = new Convolution;
let conv2 = new Convolution;
let mxPool1 = new MaxPool;

let set = mnist.set(8000, 2000)
let filter_count_1 = 4
let filter_count_2 = 32

let f = []
for(let i = 0; i<filter_count_1; i++) f.push(createMatrix(1,5,5,()=>Math.random()))

let f2 = []
for(let i= 0; i<filter_count_2; i++) f2.push(createMatrix(filter_count_1,5,5,()=>Math.random()))

let network = new NeuralNetwork({
  input_nodes : 8*8*filter_count_2,
  layer_count : [100,25],
  output_nodes :10,
  weight_bias_initilization_range : [-1,1]
});
network.Activation.hidden = [(x)=>1/(1+Math.exp(-x)),(x)=>x*(1-x)]
let acc = {t:0,f:0}
let Data = [
    {
        cost:[],
        pred:[],
        true:[],
        step:[],
        acc:[]
    }
]
let BATCH_SIZE = 1
let EPOCH = 1
let BATCH_Stack = {
    Filters:[],
    FullyConnected:[]
}
let cost = 0
function forward_pass(input,desired){
    // get input and draw
    // drawMatrix(input,"inp")
    //pool1 and draw
    let y1 = conv.convolution(input,f,true)
    if(y1.flat(Infinity).filter(e=>e===0).length != y1.flat(Infinity).length)
      y1 = La.normalize(y1,-1,1)
    // conv.drawFilters("Filter_img","f1")
    // drawMatrix(y1,"y1")
    //convolution and draw
    let y2 = mxPool1.pool(y1)
    // drawMatrix(y2,"y2")
    //conv2 and draw
    let y3 = conv2.convolution(y2,f2,true)
    if(y3.flat(Infinity).filter(e=>e===0).length != y3.flat(Infinity).length)
      y3 = La.normalize(y3,0,1)
    // drawMatrix(y3,"y3")
    // conv2.drawFilters("Filter_img","f2")

    // drawMatrix(conv2.F,"f2")
    //feed to network and get cost
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
for(let epoch = 0; epoch < EPOCH; epoch++){
    acc = {t:0,f:0}
    let bar1 = new cliProgress.SingleBar({
        format: 'Epoch:{epoch} [{bar}] {percentage}% | ETA: {eta}s | {value}/{total} | Acc: {acc}% | Loss: {loss}'
    }, cliProgress.Presets.shades_classic);
    bar1.start(set.training.length, 0,{
        epoch:epoch,
        acc:acc.t*BATCH_SIZE/10,
        loss:0
    })
    for(let step = 0; step < set.training.length; step++){
        //------forward pass------//
        let x = La.normalize(set.training[step].input,-1,1)
        x = La.reconstructMatrix(x,{x:28,y:28,z:1})
        let desired = set.training[step].output

        let [y1,y2,y3,out, pred] = forward_pass(x,desired)
        cost = out.Cost.toFixed(3)
        //------backword pass------//

        if(step%1000 == 0){
            acc = {t:0,f:0}
            fs.writeFileSync('logs\\Info.json', JSON.stringify(Data));
        }

        if( step%BATCH_SIZE == 0){
            if(step != 0){
                let WeightUpdate =  BATCH_Stack.FullyConnected[0].WeightUpdate
                let BiasUpdate =  BATCH_Stack.FullyConnected[0].BiasUpdate
                let F1 = BATCH_Stack.Filters[0].grads_y1
                let F2 = BATCH_Stack.Filters[0].grads_y3
                for(let i = 1; i< BATCH_SIZE; i++){
                    WeightUpdate = add(WeightUpdate,BATCH_Stack.FullyConnected[i].WeightUpdate)
                    BiasUpdate = add(BiasUpdate,BATCH_Stack.FullyConnected[i].BiasUpdate)
                    F1 = add(F1,BATCH_Stack.Filters[i].grads_y1)
                    F2 = add(F2,BATCH_Stack.Filters[i].grads_y3)
                }
                // console.log( Math.max(...WeightUpdate.flat(Infinity) )  )
                // console.log( Math.max(...BiasUpdate.flat(Infinity) )  )
                // console.log( Math.max(...F1.flat(Infinity) )  )
                // console.log( Math.max(...F2.flat(Infinity) )  )
                network.update(WeightUpdate,BiasUpdate,0.01);
                conv2.filterGrads(F2,1e-3)
                conv2.F = La.normalize(conv2.F,-1,1)
                conv.filterGrads(F1,1e-3)
                conv.F = La.normalize(conv.F,-1,1)

                if(pred.indexOf(Math.max(...pred))==desired.indexOf(Math.max(...desired))){
                    acc.t += 1
                }else{
                    acc.f += 1
                }
                if (Data[epoch] === undefined) Data[epoch] = {cost:[],pred:[],true:[],step:[],acc:[]};

                Data[epoch].cost.push(parseFloat(out.Cost.toFixed(3)))
                Data[epoch].pred.push(pred.indexOf(Math.max(...pred)))
                Data[epoch].true.push(desired.indexOf(Math.max(...desired)))
                Data[epoch].step.push(step)
                Data[epoch].acc.push(parseFloat((acc.t/(1000/BATCH_SIZE)).toFixed(3)))
            }
            BATCH_Stack.Filters = []
            BATCH_Stack.FullyConnected = []
            bar1.increment(BATCH_SIZE,{epoch:epoch,loss:out.Cost.toFixed(3),acc:acc.t*BATCH_SIZE/10})
        }
        BATCH_Stack.Filters.push(backword_pass())
        BATCH_Stack.FullyConnected.push({WeightUpdate:network.WeightUpdates,BiasUpdate:network.BiasUpdates})
    }
    bar1.stop()
    console.log(
        "cost:",cost,
        "epoch",epoch,
        "acc:",(acc.t/acc.f).toFixed(3),acc.t/(1000/BATCH_SIZE),acc.f/(1000/BATCH_SIZE)
    );
    fs.writeFileSync('logs\\Info.json', JSON.stringify(Data));
    network.save("Net1")
    conv.saveFilters("Conv")
    conv2.saveFilters("Conv2")
    mxPool1.savePool("Pool")
}
