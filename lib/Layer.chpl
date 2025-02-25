module Layer {
    private use Tensor;
    private use Network;
    private use Env;

    class ReLU : Module(?) {

        proc init(type eltType = defaultEltType) do
            super.init(eltType);

        override proc forward(input: dynamicTensor(eltType)): dynamicTensor(eltType) do
            return input.relu();

        override proc attributes(): moduleAttributes do
            return moduleAttributes("ReLU",moduleName);
    }

    class GELU : Module(?) {

        proc init(type eltType = defaultEltType) do
            super.init(eltType);

        override proc forward(input: dynamicTensor(eltType)): dynamicTensor(eltType) do
            return input.gelu();

        override proc attributes(): moduleAttributes do
            return moduleAttributes("GELU",moduleName);
    }




    class ResidualBlock : Module(?) {
        var innerModule : Module(eltType);

        proc init(innerModule: Module(?)) {
            super.init(innerModule.eltType);
            this.innerModule = innerModule;
        }

        override proc forward(input: dynamicTensor(eltType)): dynamicTensor(eltType) do
            return this.innerModule(input) + input;
    }

}