use VGG;
use Tensor;

config param vggExampleDir = ".";

writeln("VGG Example Directory: ", vggExampleDir);

config const k = 5;
config const modelDir = vggExampleDir + "/models/vgg16/";
config const labelFile = vggExampleDir + "/imagenet/LOC_synset_mapping.txt";


proc getLabels(): [] {
  use IO;
  const r = openReader(labelFile);
  const lines = r.lines(stripNewline=true);
  // for each line, split on space, take the second part
  forall l in lines {
    var splat = l.split(" ", maxsplit=1);
    l = splat[1];
  }
  return lines;
}

// proc confidence(x: ndarray(2,real(32))): ndarray(1,real(32)) {
proc confidence(x: ndarray(1,real(32))) {
    use Math;
    var expSum = + reduce exp(x.data);
    const res = (exp(x.data) / expSum) * 100.0;
    return new ndarray(res);
    // const X: ndarray(1,real(32)) = x.squeeze(1);
    // const (_,i) = x.shape;
    // const smX = x.softmax();
    // return smX;


}

// returns (top k indicies, top k condiences)
proc run(model: shared VGG16(real(32)), file: string) {


    // writeln("Loading image: ", file);
    // // const image: dynamicTensor(real(32)) = dynamicTensor.loadImage(imagePath=file,eltType=real(32));
    // const imageData: ndarray(3,real(32)) = ndarray.loadImage(imagePath=file,eltType=real(32));
    // writeln("Loaded image: ", file);
    // writeln("Image shape: ", imageData.shape);
    // const image: dynamicTensor(real(32)) = imageData.toTensor(); // new dynamicTensor(imageData);
    // writeln("Converted image to dynamicTensor (or Tensor).");

    // writeln("Running model on image.");
    // const output: dynamicTensor(real(32)) = model(image);
    // writeln("Output shape: ", output.shape());
    // writeln("Output type: ", output.type:string);

    // const img = Tensor.load(file):real(32);
    // const imageData: ndarray(3,real(32)) = ndarray.loadImage(imagePath=file,eltType=real(32));
    const imageData = ndarray.loadFrom(file,3,real(32));
    const img = new dynamicTensor(imageData);

    writeln("imageData shape: ", imageData.shape);
    writeln("img shape: ", img.shape());
    const output = model(img);


    // const predictions: ndarray(1,real(32)) = output.forceRank(rank=1).array;
    // const percent = confidence(predictions);
    
    // const topPredictions: ndarray(1,int) = predictions.topk(k);
    // var percentTopk = [i in 0..<k] percent[topPredictions[i]];
    // return (topPredictions.data, percentTopk);

    const top = output.topk(k);
    var topArr = top.tensorize(1).array.data;
    var percent = confidence(output.tensorize(1).array);

    var percentTopk = [i in 0..<k] percent(topArr[i]);
    return (topArr, percentTopk);



    // const imageData: ndarray(3,real(32)) = ndarray.loadImage(imagePath=file,eltType=real(32));
    // writeln("Loaded image: ", file);
    // writeln("Image shape: ", imageData.shape);
    // const img = imageData.toTensor();
    // const output = model(img);

    // const top = output.topk(k);
    // var topArr = top.tensorize(1).array.data;
    // var percent = confidence(output.tensorize(1).array.data);

    // var percentTopk = [i in 0..<k] percent(topArr[i]);
    // return (topArr, percentTopk);

}

import Path;

proc runX(file: string) {


    writeln("Loading image: ", file);
    // const image: dynamicTensor(real(32)) = dynamicTensor.loadImage(imagePath=file,eltType=real(32));
    const imageData: ndarray(3,real(32)) = ndarray.loadImage(imagePath=file,eltType=real(32));
    writeln("Loaded image: ", file);
    writeln("Image shape: ", imageData.shape);
    const image = imageData.toTensor().forceRank(3).array.reshape(1,3,720,1280); // new dynamicTensor(imageData);
    writeln("Converted image to dynamicTensor (or Tensor).");

    writeln("Running model on image.");

    const output = image.loadRunModel(2, vggExampleDir + "/models/trace_vgg16.pt");
    writeln("Output shape: ", output.shape);
    writeln("Output type: ", output.type:string);

    const predictions: ndarray(2,real(32)) = output;
    const percent = confidence(predictions);
    
    const topPredictions: ndarray(1,int) = predictions.squeeze(1).topk(k);
    var percentTopk = [i in 0..<k] percent[topPredictions[i]];
    return (topPredictions.data, percentTopk);
}


config const runNewVgg = false;

proc main(args: [] string) {


    // const a = ndarray.loadPyTorchTensorDictWithKey(2,vggExampleDir + "/my_tensor_dict.pt","a");
    // const b = ndarray.loadPyTorchTensorDictWithKey(2,vggExampleDir + "/my_tensor_dict.pt","b");
    // writeln("a sum: ", a.sum());
    // writeln("b sum: ", b.sum());

    writeln("Loading labels from ", labelFile);
    const labels = getLabels();
    writeln("Loaded ", labels.size, " labels.");

    writeln("Constructing VGG16 model.");
    const vgg = new shared VGG16(real(32));
    writeln("Constructed VGG16 model.");

    writeln("Loading VGG16 model weights.");
    vgg.loadPyTorchDump(modelDir, false);
    writeln("Loaded VGG16 model.");


    var files = args[1..];

    for f in files {
        var (topArr, percent) = run(vgg, f);
        writeln("For '", f, "' the top ", k, " predictions are: ");
        for i in 0..<k {
        writef("  %?: label=%?; confidence=%2.2r%%\n", i, labels[topArr[i]], percent[i]);
        }
        writeln();
    }
}
