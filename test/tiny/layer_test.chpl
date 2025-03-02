use Tensor;
use Layer;
use Network except ReLU, Linear, Flatten;


var x = Tensor.arange(2,3);
writeln(x);

var relu = new ReLU();

// var y = relu(x);
// writeln(y);

// writeln(relu.signature);

var net = new Sequential(
    new shared ReLU()
);
var y = net(x);
writeln(y);
writeln(net.signature);


var net2 = new Sequential(
    new shared ReLU(),
    new shared Flatten(),
    new shared Linear(6,6),
    new shared ReLU(),
    new shared GELU(),
    new shared ELU()
);

y = net2(x);
writeln(y);
writeln(net2.signature);