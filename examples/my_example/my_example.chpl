use Tensor;


proc main() {
    writeln("Hello, world!");

    var a = ndarray.arange(3, 3);
    writeln("a: ", a);

    var b = ndarray.arange(3, 3);
    writeln("b: ", b);

    var c = ndarray.matmul(a, b);
    writeln("c: ", c);


}