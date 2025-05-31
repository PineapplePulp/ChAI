use Tensor;
use Layer;
import Utilities as util;

config const cpuScale: real = 0.2;
config const accelScale: real = 0.8;
config const debugCPUOnly: bool = false;


const cpuScaleFactor = cpuScale;
const acceleratorScaleFactor = accelScale;


export proc getScaledFrameWidth(width: int): int do
    if Env.acceleratorAvailable() then
        return (width:real * acceleratorScaleFactor):int;
    else
        return (width:real * cpuScaleFactor):int;

export proc getScaledFrameHeight(height: int): int do
    if Env.acceleratorAvailable() then
        return (height:real * acceleratorScaleFactor):int;
    else
        return (height:real * cpuScaleFactor):int;


// if debugCPUOnly then
//     writeln("Debugging CPU only!");
// Env.debugCpuOnlyMode(debugCPUOnly);

writeln("CPU Scale Factor: ", cpuScaleFactor);
writeln("Accelerator Scale Factor: ", acceleratorScaleFactor);
writeln("Accelerator Available: ", Env.acceleratorAvailable());


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
config var isStyleTransferModel: bool = modelPath != "sobel.pt";

var modelLayer : shared TorchModule(real(32))?;


export proc globalLoadModel() do
    modelLayer = new shared LoadedTorchModel(modelPath);


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

    lastFrame = getTime();


    const shape = (height,width,channels);
    const frameDom = util.domainFromShape((...shape));

    var ndframe = new ndarray(real(32),shape);
    ref ndframeData = ndframe.data;
    forall idx in frameDom do
        ndframeData[idx] = frame[util.linearIdx(shape,idx)];
    const dtInput = new dynamicTensor(ndframe);
    const dtOutput = if isStyleTransferModel 
                        then modelLayer!.forward(dtInput) / 255.0
                        else modelLayer!.forward(dtInput);
    const outputFrame = dtOutput.flatten().toArray(1);
    return outputFrame;
}


