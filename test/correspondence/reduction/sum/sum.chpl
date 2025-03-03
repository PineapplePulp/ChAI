use Tensor;

var a = staticTensor.arange(2,3);

Testing.numericPrint(a.sum(0));

Testing.numericPrint(a.sum(1));

Testing.numericPrint(a.sum());

var b = staticTensor.arange(2,3,4);

Testing.numericPrint(b.sum(0));

Testing.numericPrint(b.sum(1));

Testing.numericPrint(b.sum(2));

Testing.numericPrint(b.sum(0,1));

Testing.numericPrint(b.sum(1,2));

Testing.numericPrint(b.sum(0,2));
