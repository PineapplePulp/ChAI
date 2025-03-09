use Tensor;

var a = ndarray.arange(2,3);

writeln(a);

writeln(a.sum(0));

writeln(a.sum(1));

writeln(a.mean(0));
writeln(a.mean(1));


var b = ndarray.arange(2,3,4);

writeln(b);

writeln(b.sum(0,1));
writeln(b.sum(1));

writeln(b.mean(0));
writeln(b.mean(1));


writeln(b.mean(0,1));
writeln(b.mean(1,2));

{
    var a = ndarray.arange(2,3);

    var i = 0;

    i += 1;
    writeln("Test ",i,": ",a.sum(0));

    i += 1;
    writeln("Test ",i,": ",a.sum(1));

    i += 1;
    writeln("Test ",i,": ",a.sum());

    var b = ndarray.arange(2,3,4);

    i += 1;
    writeln("Test ",i,": ",b.sum(0));

    i += 1;
    writeln("Test ",i,": ",b.sum(1));

    i += 1;
    writeln("Test ",i,": ",b.sum(2));

    i += 1;
    writeln("Test ",i,": ",b.sum(0,1));

    i += 1;
    writeln("Test ",i,": ",b.sum(1,2));

    i += 1;
    writeln("Test ",i,": ",b.sum(0,2));
}

// proc foo(args: int...?nargs) {
//     writeln("args: ",args);
//     writeln("nargs: ",nargs);
// }

// foo(1,2,3);
// foo()

{
    var b = staticTensor.arange(2,3,4);
    writeln(b.mean(0));
    writeln(b.mean(1));
    writeln(b.mean(0,1));
}