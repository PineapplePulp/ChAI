use Tensor;

var a = staticTensor.arange(2,3);

Testing.numericPrint(a.mean(0));

Testing.numericPrint(a.mean(1));

Testing.numericPrint(a.mean());

var b = staticTensor.arange(2,3,4);

Testing.numericPrint(b.mean(0));

Testing.numericPrint(b.mean(1));

Testing.numericPrint(b.mean(2));

Testing.numericPrint(b.mean(0,1));

Testing.numericPrint(b.mean(1,2));

Testing.numericPrint(b.mean(0,2));
