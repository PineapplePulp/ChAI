use Tensor;


proc main(args: [] string) {
    // const a = ndarray.loadPyTorchTensorDictWithKey(2,"models/my_tensor_dict.pt","a");
    // const b = ndarray.loadPyTorchTensorDictWithKey(2,"models/my_tensor_dict.pt","b");
    // writeln("a sum: ", a.sum());
    // writeln("b sum: ", b.sum());

    var image = ndarray.loadFrom(args[1],3,real(32));

    image = image.resize(224,224).imageNetNormalize();
    writeln("Resized image: ", image.shape);

    var batchedImage = ndarray.loadFrom(args[1],3,real(32)).unsqueeze(0);
    writeln("Batched image: ", batchedImage.shape);

    // batchedImage = batchedImage.resize(224,224);
    writeln("Batched image resized: ", batchedImage.shape);

    image = batchedImage.squeeze(3).imageNetNormalize();
    writeln("Squeezed image: ", image.shape);

    image.saveImage("test.jpg");
}