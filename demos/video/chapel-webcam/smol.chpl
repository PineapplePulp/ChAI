
export proc square(x: int): int {
    writeln(x, " * ", x, " = ", x * x);
    return x * x;
}

export proc sumArray(a: [] int): int {
    var sum: int = 0;
    for x in a do
        sum += x;
    return sum;
}

export proc printArray(a: [] int): void {
    writeln(a);
}

