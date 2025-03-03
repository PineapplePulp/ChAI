use Tensor;

var a = Tensor.zeros(2,3).softplus();
Testing.numericPrint(a);

var b = (Tensor.zeros(2,3,4) - 60.0).softplus();
Testing.numericPrint(b);

var c = (Tensor.zeros(10) + 40.0).softplus();
Testing.numericPrint(c);

// same values with beta = -0.001
a = Tensor.zeros(2,3).softplus(beta=-0.001);
Testing.numericPrint(a);

b = (Tensor.zeros(2,3,4) - 60.0).softplus(beta=-0.001);
Testing.numericPrint(b);

c = (Tensor.zeros(10) + 40.0).softplus(beta=-0.001);
Testing.numericPrint(c);

// same values with beta = 10.0
a = Tensor.zeros(2,3).softplus(beta=10.0);
Testing.numericPrint(a);

b = (Tensor.zeros(2,3,4) - 60.0).softplus(beta=10.0);
Testing.numericPrint(b);

c = (Tensor.zeros(10) + 40.0).softplus(beta=10.0);
Testing.numericPrint(c);

// beta = 0 should return infinity in all values
a = Tensor.zeros(5).softplus(beta=0.0);
Testing.numericPrint(a);