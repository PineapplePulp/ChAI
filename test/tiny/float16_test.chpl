use Tensor;

var f16 = dynamicTensor.load("f16.chdata",dtype=real(32),debug=true);
writeln(f16);

var f32 = dynamicTensor.load("f32.chdata",dtype=real(32),debug=true);
writeln(f32);

var f64 = dynamicTensor.load("f64.chdata",dtype=real(64),debug=true);
writeln(f64);