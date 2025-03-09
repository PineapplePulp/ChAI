use Tensor;
use Network;

var m = new BatchNorm(
    num_features=3
);
m.setup();

for idx in m.movingAvg.array(1).domain {
    m.movingAvg.array(1)[idx] = 0;
}

for idx in m.movingVar.array(1).domain {
    m.movingVar.array(1)[idx] = 2;
}

// .data because these are Parameters.
for idx in m.weight.data.array(1).domain {
    m.weight.data.array(1)[idx] = 4;
}

for idx in m.bias.data.array(1).domain {
    m.bias.data.array(1)[idx] = 1;
}

var x1: Tensor(real) = Tensor.arange(2, 3);
const output1 = m(x1);
writeln("2D Test: ", output1);

// var x2 = Tensor.arange(2, 3, 3);
// const output2 = m(x2);
// writeln("3D Test: ", output2);

// var x3 = Tensor.arange(2, 3, 2, 2);
// const output3 = m(x3);
// writeln("4D Test: ", output3);