
// require "mylib.h", "-lMyLib";

extern proc baz(): int;
extern proc wrHello(): void;
extern proc wrHelloTorch(): void;
extern proc sumArray(arr: [] real(32), sizes: [] int(32), dim: int(32)): real(32);
extern proc increment(arr: [] real(32), sizes: [] int(32), dim: int(32), ref output: [] real(32)): void;

extern record tensor_result_t {
    var data: c_ptr(real(32));
    var sizes: c_ptr(int(32));
    var dim: int(32);
}

extern proc increment2(arr: [] real(32), sizes: [] int(32), dim: int(32)): tensor_result_t;


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


record arrayShape_c {
    param rank: int;
    var sizes: [0..<rank] int(32);
}

proc getSizeArray(const ref arr: [] ?eltType): [] int(32) {
    var sizes: [0..<arr.rank] int(32);
    for i in 0..<arr.rank do
        sizes[i] = arr.dim(i).size : int(32);
    return sizes;
}

proc getArrayShapeC(const ref arr: [] ?eltType): arrayShape_c(arr.rank) {
    var shape: arrayShape_c(arr.rank);
    for i in 0..<arr.rank do
        shape.sizes[i] = arr.dim(i).size : int(32);
    return shape;
}

// writeln("Sum of array: ", sumArray(a,getSizeArray(a),a.rank));

// var shape = getArrayShapeC(a);
// writeln("Shape of array: ", shape.sizes);
// writeln("Sum of array: ", sumArray(a,shape.sizes,shape.rank));

var shape = getArrayShapeC(a);
writeln("A: ", a);

var b: [a.domain] real(32);
increment(a,shape.sizes,shape.rank,b);
writeln("B: ", b);


proc getResultTensorShape(param dim: int, result: tensor_result_t): dim*int {
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

var c = increment2(a,shape.sizes,shape.rank);

var cShape = getResultTensorShape(shape.rank, c);

var cDom = domainFromShape((...cShape));

var C: [cDom] real(32);
forall i in 0..<cDom.size {
    var idx = cDom.orderToIndex(i);
    C[idx] = c.data[i];
}


writeln("C: ", C);

