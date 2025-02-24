use Tensor;
use Layer;


var x = Tensor.arange(2,3);
writeln(x);

var relu = new Activation("relu");

writeln("Debug: ",relu.activationName);
var y = relu(x);
writeln(y);