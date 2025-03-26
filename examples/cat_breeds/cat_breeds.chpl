use Tensor;
use Network except Linear, Flatten, ReLU;
use Layer;

class SmallCNN : Module(?) {
    var layers: shared Sequential(?);
    proc init(type eltType = defaultEltType) {
        this.layers = new shared Sequential(
            new shared Conv2D(eltType,channels=3,features=64,kernel=3,stride=1,padding=1,bias=false),
            new shared ReLU(),
            new shared Conv2D(eltType,channels=64,features=128,kernel=3,stride=1,padding=1,bias=false),
            new shared ReLU(),
            new shared MaxPool(eltType,2),
            new shared Flatten(eltType),
            new shared Linear(eltType,8192,256),
            new shared Linear(eltType,256,10)
        );

    }
    override proc forward(input: dynamicTensor(eltType)): dynamicTensor(eltType) do
        return this.layers.forward(input);
}

