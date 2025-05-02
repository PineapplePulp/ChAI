use Tensor;


proc main() {
    writeln("Hello, world!");

    var a: ndarray(2,real(32)) = ndarray.arange(3, 3);
    writeln("a: ", a);

    var b: ndarray(2,real(32)) = ndarray.arange(3, 3);
    writeln("b: ", b);

    var c = ndarray.addTwoArrays(a,b);
    writeln("a + b = c: ", c);


}