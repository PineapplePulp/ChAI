use Tensor;
use Layer;
use Network except ReLU;


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