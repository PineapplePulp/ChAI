use Tensor;


proc main() {
    const a = ndarray.loadPyTorchTensorDictWithKey(2,"models/my_tensor_dict.pt","a");
    const b = ndarray.loadPyTorchTensorDictWithKey(2,"models/my_tensor_dict.pt","b");
    writeln("a sum: ", a.sum());
    writeln("b sum: ", b.sum());
}