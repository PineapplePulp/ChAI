
// require "mylib.h", "-lMyLib";

extern proc baz(): int;
extern proc wrHello(): void;
extern proc wrHelloTorch(): void;
extern proc sumArray(arr: [] real(32), sizes: [] int(32), dim: int(32)): real(32);

wrHello();

wrHelloTorch();

writeln("baz: ", baz());


var dom = {0..<10, 0..<10};
var a: [dom] real(32);
for (idx,i) in zip(dom,0..<dom.size) do
    a[idx] = i:real(32);

var sizes: [0..1] int(32);
sizes[0] = dom.dim(0).size : int(32);
sizes[1] = dom.dim(1).size : int(32);

writeln("Sum of array: ", sumArray(a,sizes,a.rank));


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

writeln("Sum of array: ", sumArray(a,getSizeArray(a),a.rank));

var shape = getArrayShapeC(a);
writeln("Shape of array: ", shape.sizes);
writeln("Sum of array: ", sumArray(a,shape.sizes,shape.rank));