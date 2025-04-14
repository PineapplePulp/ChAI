use Tensor;
use Layer;
use Network except ReLU, Linear, Flatten;

use Bridge;
use Utilities as util;

var x = Tensor.arange(3,10,10);
writeln(x);

var relu = new ReLU();

// var y = relu(x);
// writeln(y);

// writeln(relu.signature);

var net = new Sequential(
    new shared ReLU()
);
// var y = net(x);
// writeln(y);
// writeln(net.signature);


var net2 = new Sequential(
    new shared Conv2D(3,32,3)
    // new shared ReLU()
    // new shared Flatten(),
    // new shared Linear(96,6),
    // new shared ReLU(),
    // new shared GELU(),
    // new shared ELU(),
    // new shared ResidualBlock(new shared ReLU())
);


var y = net2(x);
writeln(y);
// writeln(net2.signature);



var dom = {0..<10, 0..<10};
var a: [dom] real(32);
for (idx,i) in zip(dom,0..<dom.size) do
    a[idx] = i:real(32);



var input: [util.domainFromShape(2,64,28,28)] real(32) = 1.0;
var kernel: [util.domainFromShape(128,64,3,3)] real(32) = 2.0;
var bias: [util.domainFromShape(128)] real(32) = 3.0;
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


var arr = ndarray.arange(2,3);
writeln(arr);
writeln(arr.toBridgeTensor());

var bt = arr.toBridgeTensor();
writeln(bt);
arr.loadFromBridgeTensor(bt);
writeln(arr);
writeln(arr.shape);
writeln(bt);

// writeln(ndarray.fromBridgeTensor(2,bt));

proc f() {
var net2 = new Sequential(
    new shared Conv2D(3,32,3),
    new shared ReLU(),
    new shared Flatten(),
    new shared Linear(2048,6)
    // new shared ReLU(),
    // new shared GELU(),
    // new shared ELU(),
    // new shared ResidualBlock(new shared ReLU())
);

var x = Tensor.arange(3,10,10);
var z = net2(x);
writeln(z);
writeln(z.shape());


}

f();