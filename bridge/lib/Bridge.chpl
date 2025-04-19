
// require "mylib.h", "-lMyLib";
use Allocators;

extern proc baz(): int;
extern proc wrHello(): void;
extern proc wrHelloTorch(): void;
extern proc sumArray(arr: [] real(32), sizes: [] int(32), dim: int(32)): real(32);
extern proc increment(arr: [] real(32), sizes: [] int(32), dim: int(32), ref output: [] real(32)): void;

extern record bridge_tensor_t {
    var data: c_ptr(real(32));
    var sizes: c_ptr(int(32));
    var dim: int(32);
}



extern proc increment2(arr: [] real(32), sizes: [] int(32), dim: int(32)): bridge_tensor_t;
extern proc increment3(in arr: bridge_tensor_t): bridge_tensor_t;

extern proc convolve2d(
    in input: bridge_tensor_t, 
    in kernel: bridge_tensor_t, 
    in bias: bridge_tensor_t, 
    in stride: int(32), 
    in padding: int(32)): bridge_tensor_t;

extern proc unsafe(const ref arr: [] real(32)): c_ptr(real(32));

// baz();

// wrHello();


// wrHelloTorch();

// writeln("baz: ", baz());


var dom = {0..<10, 0..<10};
var a: [dom] real(32);
for (idx,i) in zip(dom,0..<dom.size) do
    a[idx] = i:real(32);

// var sizes: [0..1] int(32);
// sizes[0] = dom.dim(0).size : int(32);
// sizes[1] = dom.dim(1).size : int(32);

// writeln("Sum of array: ", sumArray(a,sizes,a.rank));


// record arrayShape_c {
//     param rank: int;
//     var sizes: [0..<rank] int(32);
// }



// proc getArrayShapeC(const ref arr: [] ?eltType): arrayShape_c(arr.rank) {
//     var shape: arrayShape_c(arr.rank);
//     for i in 0..<arr.rank do
//         shape.sizes[i] = arr.dim(i).size : int(32);
//     return shape;
// }

// writeln("Sum of array: ", sumArray(a,getSizeArray(a),a.rank));

// var shape = getArrayShapeC(a);
// writeln("Shape of array: ", shape.sizes);
// writeln("Sum of array: ", sumArray(a,shape.sizes,shape.rank));

// var shape = getArrayShapeC(a);
// writeln("A: ", a);

// var b: [a.domain] real(32);
// increment(a,shape.sizes,shape.rank,b);
// writeln("B: ", b);

// var c = increment2(a,shape.sizes,shape.rank);

// var cShape = getResultTensorShape(shape.rank, c);

// var cDom = domainFromShape((...cShape));

// var C: [cDom] real(32);
// forall i in 0..<cDom.size {
//     var idx = cDom.orderToIndex(i);
//     C[idx] = c.data[i];
// }

// var c = bridgeTensorToArray(shape.rank, increment2(a,shape.sizes,shape.rank));


// writeln("C: ", c);

proc getSizeArray(const ref arr: [] ?eltType): [] int(32) {
    var sizes: [0..<arr.rank] int(32);
    for i in 0..<arr.rank do
        sizes[i] = arr.dim(i).size : int(32);
    return sizes;
}

proc bridgeTensorShape(param dim: int, result: bridge_tensor_t): dim*int {
    var shape: dim*int;
    for i in 0..<dim do
        shape[i] = result.sizes[i] : int;
    return shape;
}

proc domainFromShape(shape: int ...?rank): domain(rank,int) {
    const _shape = shape;
    var ranges: rank*range;
    for param i in 0..<rank do
        ranges(i) = 0..<_shape(i);
    return {(...ranges)};
}


proc bridgeTensorToArray(param rank: int, package: bridge_tensor_t): [] real(32) {
    var shape = bridgeTensorShape(rank, package);
    var dom = domainFromShape((...shape));
    var result: [dom] real(32);
    forall i in 0..<dom.size {
        var idx = dom.orderToIndex(i);
        result[idx] = package.data[i];
    }
    deallocate(package.data);
    deallocate(package.sizes);
    return result;
}


proc createBridgeTensor(const ref data: [] real(32)): bridge_tensor_t {
    var result: bridge_tensor_t;
    result.data = c_ptrToConst(data) : c_ptr(real(32));
    result.sizes = allocate(int(32),data.rank);
    const sizeArr = getSizeArray(data);
    for i in 0..<data.rank do
        result.sizes[i] = sizeArr[i];

    result.dim = data.rank;
    return result;
}

// proc createBridgeTensor(const ref data: [] real(32)): bridge_tensor_t_const {
//     var result: bridge_tensor_t_const;
//     result.data = c_ptrToConst(data);
//     result.sizes = allocate(int(32),data.rank);
//     const sizeArr = getSizeArray(data);
//     for i in 0..<data.rank do
//         result.sizes[i] = sizeArr[i];

//     result.dim = data.rank;
//     return result;
// }


proc chplIncrement(ref data: [] real(32)): [] real(32) {
    param rank = data.rank;
    var dataBT = createBridgeTensor(data);
    var resultBT = increment3(dataBT);
    var result = bridgeTensorToArray(rank, resultBT);
    deallocate(dataBT.data);
    deallocate(dataBT.sizes);
    return result;
}



// writeln(bridgeTensorToArray(2,increment3(createBridgeTensor(a))));
writeln(a);
writeln("----------");
writeln(chplIncrement(a));
writeln("----------");
writeln(a);


var input: [domainFromShape(2,64,28,28)] real(32) = 1.0;
var kernel: [domainFromShape(128,64,3,3)] real(32) = 2.0;
var bias: [domainFromShape(128)] real(32) = 3.0;
var stride: int(32) = 1;
var padding: int(32) = 1;
writeln("Begin.");
var resultBT = convolve2d(createBridgeTensor(input), createBridgeTensor(kernel), createBridgeTensor(bias), stride, padding);
var result = bridgeTensorToArray(4, resultBT);
// writeln("Input: ", input);
// writeln("Kernel: ", kernel);
// writeln("Bias: ", bias);
// writeln("Result: ", result);
writeln("Result: ", result.size);


