use Tensor;


proc test(c: ?scalarType,a: staticTensor(?rank,?eltType)) 
        where isNumericType(scalarType) {
    var C = staticTensor.valueLike(a,c : eltType);

    writeln(a + c == a + C);
    writeln(a - c == a - C);
    writeln(a * c == a * C);
    writeln(a / c == a / C);

    writeln(c + a == C + a);
    writeln(c - a == C - a);
    writeln(c * a == C * a);
    writeln(c / a == C / a);

}

proc test(c: ?scalarType,a: dynamicTensor(?eltType)) 
        where isNumericType(scalarType) {
    var C = dynamicTensor.valueLike(a,c : eltType);

    writeln(a + c == a + C);
    writeln(a - c == a - C);
    writeln(a * c == a * C);
    writeln(a / c == a / C);

    writeln(c + a == C + a);
    writeln(c - a == C - a);
    writeln(c * a == C * a);
    writeln(c / a == C / a);
}

{
    var a = tensor.arange(2,3);
    writeln(a);
    test(1.5,a);
}

{
    var a = dynamicTensor.arange(2,3);
    writeln(a);
    test(1.5,a);
}
