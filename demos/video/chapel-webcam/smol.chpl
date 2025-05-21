import Utilities as utils;

use NDArray;
import Bridge;

export proc square(x: int): int {
    writeln(x, " * ", x, " = ", x * x);
    return x * x;
}

// export proc sumArray(a: [] int): int {
//     var sum: int = 0;
//     for x in a do
//         sum += x;
//     return sum;
// }

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

// ../style-transfer/models/exports/mps/nature_oil_painting_ep4_bt4_sw1e10_cw_1e5_float32.pt
// ../style-transfer/models/exports/mps/udnie_float32.pt
config const modelPath: string = "../style-transfer/models/exports/mps/starry_ep3_bt4_sw1e11_cw_1e5_float32.pt";
var model : Bridge.bridge_pt_model_t;

use CTypes;

export proc globalLoadModel() {
    const fpPtr: c_ptr(uint(8)) = c_ptrToConst(modelPath) : c_ptr(uint(8));
    model = Bridge.load_model(fpPtr);

    // const fpPtr: c_ptr(uint(8)) = c_ptrToConst(modelPath) : c_ptr(uint(8));
    // var model = Bridge.load_model(fpPtr);
}


var lastFrame = startTime;

export proc getNewFrame(ref frame: [] real(32),height: int, width: int,channels: int): [] real(32) {

    const t = getTime() - startTime;
    const dt = getTime() - lastFrame;
    writeln("FPS: ", 1.0 / dt);
    const shape = (height,width,channels);
    const frameDom = utils.domainFromShape((...shape));

    var btFrame: Bridge.bridge_tensor_t = Bridge.createBridgeTensorWithShape(frame,shape);
    var bt: Bridge.bridge_tensor_t;
    if modelPath == "sobel.pt" then
        bt = Bridge.model_forward(model,btFrame);
    else
        bt = Bridge.model_forward_style_transfer(model,btFrame);
    


    const nextNDFrame = bt : ndarray(3, real(32));
    const flattenedNextFrame = nextNDFrame.flatten().data;
    lastFrame = getTime();
    return flattenedNextFrame;

    // forall i in 0..<frame.size {
    //     const idx = utils.indexAt(i,(...shape));
    //     const (h,w,c) = idx;
    //     const (u,v) = (h:real(64)/height,w:real(64)/width);
    //     ref color = frame[i];
    //     // if h < width {
    //     //     frame[utils.linearIdx(shape,(h,w,c))] = frame[utils.linearIdx(shape,(h,w,c-1))];
    //     // }
    //     // if h < width {
    //     //     frame[utils.linearIdx(shape,(h,w,0))] *= Math.sin(2.0*t + 5.0 * u) : real(32);
    //     // }
    //     color *= (Math.abs(Math.sin(2.0*t + 5.0 * v)) * Math.abs(Math.sin(2.0*t + 5.0 * u))) : real(32);
    // }
    // return frame;
}


// export proc getNewFrame(frame: [] uint(8),height: int, width: int,channels: int): [] uint(8) {
//     const ret = frame;
//     return ret;
// }
