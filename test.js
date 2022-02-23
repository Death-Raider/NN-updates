class LinearAlgebra {
  constructor(){}
  basefunc = (a,b,opt) => a instanceof Array ? a.map((c, i) => this.basefunc(a[i], Array.isArray(b)?b[i]:b, opt)) : opt(a,b); // base function for any depth code
  transpose = m => m[0].map((e,i) => m.map(row => row[i])); //only depth 2
  scalarMatrixProduct = (s,m) => this.basefunc(m,s,(arr,scalar)=>arr*scalar); //max any depth
  scalarVectorProduct = (s,v) => v1.map(e=>e*s);//only depth 1
  vectorDotProduct = (v1,v2) => v1.map((e,i,a)=>e*v2[i]).reduce((a,b)=>a+b); //only both depth 1
  vectorMatrixProduct = (v,m) => v.map((e,i)=>this.scalarMatrixProduct(e,this.transpose(m)[i])).reduce((a,b)=>a.map( (x, i)=> x + b[i] )); //only depth 1 and 2
  minor = (m,i=0,j=0,s=m.length-1) => Array(s).fill(0).map((e,p)=>{
      console.log(s);
      let l = m[p+(p>=i?1:0)].slice()
      l.splice(j,1)
      return l
    })
  determinant = (m,s=m.length) =>{ // matrix (nxn) and its order n > 1
    if(s == 2){
      return m[0][0]*m[1][1] - m[0][1]*m[1][0] //determinant of 2x2 matrix
    }else{
      let sum = 0
      for(let i = 0; i < s; i++)
        sum += (-1)**(i)*m[0][i]*this.determinant(this.minor(m,0,i),s-1)
      return sum
    }
  };
  invertMatrix = (m,s=m.length) => { // any nxn matrix
    let cofactorMatrix = Array(s).fill(0).map(e=>Array(s));
    let det = 0;
    for(let i = 0; i < s; i++){
      for(let j = 0; j < s; j++)
        cofactorMatrix[j][i] = (-1)**(i+j)*this.determinant(this.minor(m,i,j),s-1); // transpose + values
      det += m[i][0]*cofactorMatrix[0][i];
    }
    if(!det){
      console.log("matrix not invertiable det =",det);
      return false
    }
    let invert =  this.scalarMatrixProduct(1/det,cofactorMatrix)
    return invert
  }
  MatrixMatrixProduct = {
    matrixProduct: (m1,m2) => m1.map(row => m2[0].map((_,i)=>this.vectorDotProduct( row, m2.map(e=>e[i]) )) ), //max depth 2
    kroneckerProduct: (a,b,r=[],t=[]) => a.map(a=>b.map(b=>a.map(y=>b.map(x=>r.push(y*x)),t.push(r=[]))))&&t, //max depth 2
  }
  weightedSum = (k=1,...M) => M.reduce((a,b)=>this.basefunc(a,b,(x,y)=>x+k*y)) ;// same but any depth
  normalize = (m,a=-1,b=1) => this.basefunc(m,{min:Math.min(...m.flat(Infinity)),max:Math.max(...m.flat(Infinity))},(x,y)=>(b-a)*(x-y.min)/(y.max-y.min)+a) ;// any depth of matrix
  vectorize = (m) => Array.isArray(m[0][0])?m.flatMap(e=>this.vectorize(e)):this.transpose(m).flat(2); // any depth
  im2row = (m,a,q=1,t=[],i=0,j=0,k=0,r) => {
    if(Array.isArray(m[0][0])){
      m.map((e,f)=>this.im2row(e,a,q,t[f]=[],i,j,f))
      for(let index = 1; index < t.length; index++)
        t[0] = t[0].map((e,i)=>e.concat(t[index][i]))
      return t[0]
    }else{
      t.push(r=[])
      for(let x = 0; x < a[1]; x++){
        for(let y = 0; y < a[0]; y++)
          r.push(m[y+i][x+j])
      }
      return ( i < m.length-a[0] ? this.im2row(m,a,q,t,i+=q,j,k) : j < m[0].length-a[1] ? this.im2row(m,a,q,t,0,j+=q,k) : t )
    }
  };
  im2col = (m,a) => this.transpose(this.im2row(m,a));
  reconstructMatrix = (flatArr,m,Matrix=[]) => {
    for(let z = 0; z < m.z; z++){
      Matrix[z] = []
      for(let i = 0; i < m.y; i++){
        Matrix[z][i] = []
        for(let j = 0; j < m.x; j++){
          Matrix[z][i][j] = flatArr[j + m.x*i + m.y*m.x*z]
        }
      }
    }
    return Matrix
  }
}
let La = new LinearAlgebra;
class MaxPool{
  constructor(){
    this.shape = []
    this.location = []
  }
  pool(inp,size = 2,stride=2,reshape=true){
    let M = inp.map((m,index)=>{
      if(m.length%stride != 0 || m[0].length%stride != 0 || size > m.length || size > m[0].length){
        console.log("pooling cant be done with set stride");
        return false;
      }
      this.shape = [inp[0].length, inp[0][0].length, inp.length]
      let H = m.length/stride;
      let W = m[0].length/stride;
      let i=0,j=0;
      this.location[index] = [];
      let phi = La.im2row(m,[size,size],stride).map((e,l)=>{
        let max = Math.max(...e);
        let q = e.indexOf(max);
        let fi = q%size+i , fj = Math.floor(q/size)+j;
        i=(i<m.length-size)? i+stride:0;
        (j < (m[0].length-size) && i==0)?j+=stride:0;
        this.location[index].push([fi,fj]);
        return max;
      })
      return La.reconstructMatrix(phi, {x:W,y:H,z:1}).map(e=>La.transpose(e)).flat();
    })
    return reshape?M:La.vectorize(M);
  }

  layerGrads(prevGrads){
    let m = Array(this.shape[1]*this.shape[0]).fill(0).map(e=>Array(this.shape[2]).fill(0).map(e=>0))
    this.location.forEach((e,i)=>{
      prevGrads.forEach((p,j)=>{ // p is location duplet
        let index = e[j][0]+e[j][1]*this.shape[1]
        m[index][i] = p[i]
      })
    })
    return m
  }

}
let max_pool = new MaxPool
let test = [
  [
    [1,2,3,1],
    [4,5,6,1],
    [7,8,9,1],
    [0,0,0,1]
  ],
  [
    [1,2,1,0],
    [2,3,6,9],
    [3,4,5,8],
    [4,5,6,7]
  ]
]

let test_grads = Array(4).fill(0).map((e,i)=>[i%2+1,(i+1)%2+1])
let q = max_pool.pool(test)
console.log(q,max_pool);
let m = max_pool.layerGrads(test_grads)
console.log(m);
