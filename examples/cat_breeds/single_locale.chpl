use Tensor;
use Network;
import Time;

config const detach = true;

Tensor.detachMode(detach); // what's this do?

writeln("Loading model...");
var model: owned Module(real(32))  = loadModel(
    specFile="./cat_breeds/models/chai_model/specification.json",
    weightsFolder = "./cat_breeds/models/chai_model/",
    dtype=real(32)
);

writeln(model.signature);

// check if this is needed
// model.loadPyTorchDump("./cat_breeds/models/chai_model/");

writeln("Loading images...");
config const numImages = 1;
var images = forall i in 0..<numImages do Tensor.load("./cat_breeds/data/catbreeds/chai_images/item"+i:string+".chdata") : real(32);

writeln("Initializing for inference...");
var preds: [0..<numImages] int;
config const numTimes = 1;
var time: real;
for i in 0..<numTimes {
    writeln("Inference (loop ",i,")...");
    var st = new Time.stopwatch();

    st.start();
    forall (img, pred) in zip(images, preds) {
        writeln(img.type:string);
        pred = model(img).argmax();
    }
    st.stop();

    const tm = st.elapsed();
    writeln("Time: ", tm, " seconds.");
}

time /= numTimes;

config const printResults = false;
if printResults {
    for i in images.domain {
        writeln((i, preds[i]));
    }
}

writeln("The average inference time for batch of size ", numImages, " was ", time, " seconds.");