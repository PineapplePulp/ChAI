use Tensor;
use Layer;
import Utilities as util;

config const cpuScale: real = 0.2;
config const accelScale: real = 0.45;
config const debugCPUOnly: bool = false;


const cpuScaleFactor = cpuScale;
const acceleratorScaleFactor = accelScale;


export proc acceleratorAvailable(): bool do
    return Bridge.acceleratorAvailable();


export proc getScaledFrameWidth(width: int): int do
    if acceleratorAvailable() then
        return (width:real * acceleratorScaleFactor):int;
    else
        return (width:real * cpuScaleFactor):int;

export proc getScaledFrameHeight(height: int): int do
    if acceleratorAvailable() then
        return (height:real * acceleratorScaleFactor):int;
    else
        return (height:real * cpuScaleFactor):int;


if debugCPUOnly then
    writeln("Debugging CPU only!");
Bridge.debugCpuOnlyMode(debugCPUOnly);

writeln("CPU Scale Factor: ", cpuScaleFactor);
writeln("Accelerator Scale Factor: ", acceleratorScaleFactor);
writeln("Accelerator Available: ", acceleratorAvailable());


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
// ../style-transfer/models/exports/mps/starry_ep3_bt4_sw1e11_cw_1e5_float32.pt // This is the one
// ../style-transfer/models/exports/cpu/mosaic_float16.pt
config const modelPath: string = "../style-transfer/models/exports/cpu/mosaic_float16.pt";
var model : Bridge.bridge_pt_model_t;

var modelLayer : shared TorchModule(real(32))?;


use CTypes;

export proc globalLoadModel() {
    const fpPtr: c_ptr(uint(8)) = c_ptrToConst(modelPath) : c_ptr(uint(8));
    model = Bridge.load_model(fpPtr);
    if modelPath == "sobel.pt" then
        modelLayer = new shared TorchModule(modelPath);
    else
        modelLayer = new shared StyleTransfer(modelPath);
}


var lastFrame = startTime;

config const chaiImpl = true;


const windowSize = 5;
var frameCount = 0;
var runningSum: real = 0;
var windowSum: real = 0;
var fpsBuffer: [0..<windowSize] real;

config const copyStrat: int = 1;


export proc getNewFrame(ref frame: [] real(32),height: int, width: int,channels: int): [] real(32) {

    const t = getTime() - startTime;
    const dt = getTime() - lastFrame;
    const fps = 1.0 / dt;

    /*
    runningSum += fps;
    frameCount += 1;
    const overallAvgFPS = runningSum / frameCount;
    const idx = (frameCount - 1) % windowSize;
    if frameCount <= windowSize {
        // still filling the buffer
        windowSum += fps;
    } else {
        // subtract the old value at this slot, then add the new one
        windowSum += fps - fpsBuffer[idx];
    }
    fpsBuffer[idx] = fps;
    const currentWindowSize = min(frameCount, windowSize);
    const windowAvgFPS = windowSum / currentWindowSize;
    writeln("FPS: ", fps, " avg FPS: ", windowAvgFPS, "max window FPS: ", max reduce fpsBuffer);
    */
    // writeln("FPS: ", fps);
    lastFrame = getTime();


    const shape = (height,width,channels);
    const frameDom = util.domainFromShape((...shape));
    // const frameArr = reshape(frame,frameDom);

    if chaiImpl {

        var ndframe = new ndarray(real(32),shape);
        ref ndframeData = ndframe.data;
        forall idx in frameDom do
            ndframeData[idx] = frame[util.linearIdx(shape,idx)];
        const dtInput = new dynamicTensor(ndframe); // 20 fps
        const dtOutput = modelLayer!.forward(dtInput);
        const outputFrame = dtOutput.flatten().toArray(1);
        return outputFrame;

        // 2 (no)
        // if copyStrat == 1 {
        //     var ndframe = new ndarray(real(32),shape);
        //     ref ndframeData = ndframe.data;
        //     forall idx in frameDom do
        //         ndframeData[idx] = frame[util.linearIdx(shape,idx)];
        //     const dtInput = new dynamicTensor(ndframe); // 20 fps
        //     const dtOutput = modelLayer!.forward(dtInput);
        //     const outputFrame = dtOutput.flatten().toArray(1);
        //     return outputFrame;
        // } else {
        //     const dtInput = (new dynamicTensor(frame)).reshape((...shape));
        //     const dtOutput = modelLayer!.forward(dtInput);
        //     const outputFrame = dtOutput.flatten().toArray(1);
        //     return outputFrame;
        // }


        // const dtInput = new dynamicTensor(reshape(frame,frameDom)); // Way slower

        // const dtOutput = modelLayer!.forward(dtInput);
        // // const outputFrame = dtOutput.rankedData(1);
        // const outputFrame = dtOutput.flatten().toArray(1);
        // return outputFrame;
    } else {
        var btFrame: Bridge.bridge_tensor_t = Bridge.createBridgeTensorWithShape(frame,shape);
        var bt: Bridge.bridge_tensor_t;
        if modelPath == "sobel.pt" then
            bt = Bridge.model_forward(model,btFrame);
        else
            bt = Bridge.model_forward_style_transfer(model,btFrame);
        
        const nextNDFrame = bt : ndarray(3, real(32));
        const flattenedNextFrame = nextNDFrame.flatten().data;
        return flattenedNextFrame;
    }

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
