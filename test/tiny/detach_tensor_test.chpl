use Tensor;

use Autograd;


var a: tensor(2,real) = tensor.arange(3,3);

writeln(a);


var b: tensor(2,real) = tensor.arange(3,3) * -0.1;
writeln(b);

var c = a + b + b + b;

writeln(c);

var d = c;

c.eraseHistory();

writeln(d.meta.showGraph());
writeln(c.meta.showGraph());

// writeln((a + b + c).meta.showGraph(indent=true));


writeln(b.gelu());
