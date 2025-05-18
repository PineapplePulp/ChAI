import Utilities as utils;

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
export proc getNewFrame(ref frame: [] real(32),height: int, width: int,channels: int): [] real(32) {
    // const dom = {0..<height,0..<width,0..<channels};
    // const rgb: [dom] real(32) = reshape(frame,dom);
    // const ret = reshape(rgb,{0..<dom.size});
    // return ret;
    // return [x in frame] x * 1.2;

    // const dom = {0..<height,0..<width,0..<channels};
    // const rgb: [dom] real(32) = reshape(frame,dom);

    // const m = max reduce frame;
    // return [x in frame] m;

    const shape = (height,width,channels);
    forall (i,color) in zip(0..<frame.size,frame) {
        const idx = utils.indexAt(i,(...shape));
        const (h,w,c) = idx;
        // const color = frame[i];
        // if h < width {
        //     frame[utils.linearIdx(shape,(h,w,c))] = frame[utils.linearIdx(shape,(h,w,c-1))];
        // }
        if h < width {
            frame[utils.linearIdx(shape,(h,w,0))] = 1.0;
        }
    }
    return frame;
}


// export proc getNewFrame(frame: [] uint(8),height: int, width: int,channels: int): [] uint(8) {
//     const ret = frame;
//     return ret;
// }
