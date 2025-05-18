
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
export proc getNewFrame(frame: [] real(32),height: int, width: int,channels: int): [] real(32) {
    const ret = frame;
    return ret;
}


// export proc getNewFrame(frame: [] uint(8),height: int, width: int,channels: int): [] uint(8) {
//     const ret = frame;
//     return ret;
// }
