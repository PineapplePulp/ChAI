use Tensor;

import Bridge;
use Utilities as util;


var x = ndarray.arange(10,10);
writeln(x);

// var y = x.toBridgeTensor();
// writeln(y);


var y = ndarray.arange(10,10);
var z = ndarray.addTwoArrays(x, y);
writeln(z);

