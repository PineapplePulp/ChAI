use Tensor;


proc main(args: [] string) {

    var image = ndarray.loadPyTorchTensor(3,args[1],real(32));
    writeln("Image shape: ", image.shape);

    image = image.resize(224,224);
    writeln("Resized image: ", image.shape);
    image = image.imageNetNormalize();
    writeln("Normed-Resized image: ", image.shape);

    var batchedImage = ndarray.loadPyTorchTensor(3,args[1],real(32)).unsqueeze(0);
    writeln("Batched image: ", batchedImage.shape);

    // batchedImage = batchedImage.resize(224,224);
    writeln("Batched image resized: ", batchedImage.shape);

    image = batchedImage.squeeze(3).imageNetNormalize();
    writeln("Squeezed image: ", image.shape);

    image.saveImage("test.jpg");
}