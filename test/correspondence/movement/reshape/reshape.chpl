use Tensor;

var a = dynamicTensor.arange(2,3);

Testing.numericPrint(a.reshape(3,2));

Testing.numericPrint(a.reshape(6));

Testing.numericPrint(a.reshape(1,2,3));

Testing.numericPrint(a.reshape(1,1,6));

Testing.numericPrint(a.reshape(6,1,1));

Testing.numericPrint(a.reshape(2,1,1,3));

Testing.numericPrint(a.reshape(3,1,1,2));

Testing.numericPrint(a.reshape(1,3,2,1));
