use Tensor;

use Autograd;

var b: Tensor(real) = Tensor.arange(3,3);

// writeln(b.sum(0));

writeln(b);
var bt = b.forceRank(2);
writeln(bt);
writeln(bt.sum(0));

var ba = bt.array;
writeln(ba);

writeln(ba.sum(0));

var baso = new sumOp(2,real,1,(0,),bt.resource);

writeln(baso.forward());


var B = bt.eraseRank();
writeln(B.sum(0));