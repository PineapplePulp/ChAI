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

proc confidence(x: []): [] {
  use Math;
  var expSum = + reduce exp(x);
  return (exp(x) / expSum) * 100.0;
}

// returns (top k indicies, top k condiences)
proc run(model: shared VGG16(real(32)), file: string) {


    writeln("Loading image: ", file);
    // const image: dynamicTensor(real(32)) = dynamicTensor.loadImage(imagePath=file,eltType=real(32));
    const imageData: ndarray(3,real(32)) = ndarray.loadImage(imagePath=file,eltType=real(32));
    writeln("Loaded image: ", file);
    writeln("Image shape: ", imageData.shape);
    const image: dynamicTensor(real(32)) = imageData.toTensor(); // new dynamicTensor(imageData);
    writeln("Converted image to dynamicTensor (or Tensor).");

    writeln("Running model on image.");
    var output: dynamicTensor(real(32)) = model(image);
    writeln("Output shape: ", output.shape());
    writeln("Output type: ", output.type:string);

    const top = output.topk(k);
    var topArr = top.forceRank(1).array.data;
    var percent = confidence(output.forceRank(1).array.data);

    var percentTopk = [i in 0..<k] percent(topArr[i]);
    return (topArr, percentTopk);
}

proc main(args: [] string) {
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
