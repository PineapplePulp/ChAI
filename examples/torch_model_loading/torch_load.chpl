use Tensor;


proc main(args: [] string) {
    // const a = ndarray.loadPyTorchTensorDictWithKey(2,"models/my_tensor_dict.pt","a");
    // const b = ndarray.loadPyTorchTensorDictWithKey(2,"models/my_tensor_dict.pt","b");
    // writeln("a sum: ", a.sum());
    // writeln("b sum: ", b.sum());

    var image = ndarray.loadFrom(args[1],3,real(32));
    writeln("Loaded image: ", args[1]);
    writeln("Image shape: ", image.shape);

    image = image.resize(224,224);
    writeln("Resized image: ", image.shape);

    image.saveImage("test.jpg");
}