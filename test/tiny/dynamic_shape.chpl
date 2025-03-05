use Tensor;

use List;
/*
var a = dynamicTensor.arange(2,3);

writeln(a);

writeln(a.shape());

writeln(a.shape().toList());



var ds = new dynamicShape((3,2));

var b = a.reshape(ds);

writeln(b);

writeln(b.shape());

writeln(b.shape().toList());
*/

var c = dynamicTensor.arange(4) + 1;
writeln(c);

var d = c.unsqueeze(0).unsqueeze(1);
writeln(d);


writeln(d.squeeze());


writeln(d.squeeze(1));

writeln(d.reshape(new dynamicShape((2,2))));
writeln(d.reshape(dShape=new dynamicShape((4,))));