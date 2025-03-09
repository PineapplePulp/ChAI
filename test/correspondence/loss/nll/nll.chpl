use Tensor;

var input = dynamicTensor.arange(2,3);
ref x = input.forceRank(2).array;
x[0,0] = -1.0; x[0,1] = -2.0; x[0,2] = -3.0;
x[1,0] = -0.5; x[1,1] = -1.5; x[1,2] = -2.5;

var target = dynamicTensor.ones(2);
ref y = target.forceRank(1).array;
y[0] = 0; y[1] = 1;

var a = dynamicTensor.nllLoss(input, target);
Testing.numericPrint(a);
