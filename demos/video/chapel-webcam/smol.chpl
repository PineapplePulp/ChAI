import Utilities as utils;

use NDArray;

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

use Time;
use Math;


proc getTime() {
    const tm = timeSinceEpoch();
    const sec = tm.chpl_seconds : real(64);
    const us = tm.chpl_microseconds : real(64);
    const t = sec + (us/1000000.0);
    return t;
}

const startTime = getTime();

export proc getNewFrame(ref frame: [] real(32),height: int, width: int,channels: int): [] real(32) {

    const t = getTime() - startTime;
    const shape = (height,width,channels);
    writeln(shape);

    var ndframe = new ndarray(real(32),shape);
    ndframe.data = reshape(frame,ndframe.domain);

    writeln(ndframe.max());
    forall i in 0..<frame.size {
        const idx = utils.indexAt(i,(...shape));
        const (h,w,c) = idx;
        const (u,v) = (h:real(64)/height,w:real(64)/width);
        ref color = frame[i];
        // if h < width {
        //     frame[utils.linearIdx(shape,(h,w,c))] = frame[utils.linearIdx(shape,(h,w,c-1))];
        // }
        // if h < width {
        //     frame[utils.linearIdx(shape,(h,w,0))] *= Math.sin(2.0*t + 5.0 * u) : real(32);
        // }
        color *= (Math.abs(Math.sin(2.0*t + 5.0 * v)) * Math.abs(Math.sin(2.0*t + 5.0 * u))) : real(32);
    }
    return frame;
}


// export proc getNewFrame(frame: [] uint(8),height: int, width: int,channels: int): [] uint(8) {
//     const ret = frame;
//     return ret;
// }
