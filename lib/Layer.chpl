module Layer {
    private use Tensor;
    private use Network;
    private use Env;

    class ReLU : Module(?) {

        proc init(type eltType = defaultEltType) {
            super.init(eltType);
            init this;
            this.moduleName = "ReLU";
        }

        override proc forward(input: dynamicTensor(eltType)): dynamicTensor(eltType) do
            return input.relu();

        override proc attributes(): moduleAttributes do
            return new moduleAttributes("ReLU",moduleName);
    }

    class GELU : Module(?) {

        proc init(type eltType = defaultEltType) {
            super.init(eltType);
            init this;
            this.moduleName = "GELU";
        }

        override proc forward(input: dynamicTensor(eltType)): dynamicTensor(eltType) do
            return input.gelu();

        override proc attributes(): moduleAttributes do
            return new moduleAttributes("GELU",moduleName);
    }

    class ELU : Module(?) {
        var alpha: eltType;

        proc init(type eltType = defaultEltType, alpha: eltType = 1.0) {
            super.init(eltType);
            this.alpha = alpha;
            init this;
            this.moduleName = "ELU";
        }

        override proc forward(input: dynamicTensor(eltType)): dynamicTensor(eltType) do
            return input.elu(alpha);

        override proc attributes(): moduleAttributes do
            return new moduleAttributes(
                "ELU",
                moduleName,
                ("alpha",alpha)
            );
    }

    class RReLU : Module(?) {
        var lower: eltType;
        var upper: eltType;

        proc init(type eltType = defaultEltType, lower: eltType = 0.125, upper: eltType = 0.333) {
            super.init(eltType);
            this.lower = lower;
            this.upper = upper;
            init this;
            this.moduleName = "RReLU";            
        }

        override proc forward(input: dynamicTensor(eltType)): dynamicTensor(eltType) do
            return input.rrelu(lower,upper);
        
        override proc attributes(): moduleAttributes do
            return new moduleAttributes(
                "RReLU",
                moduleName,
                ("lower",lower),
                ("upper",upper)
            );
    }

    class Flatten : Module(?) {
        proc init(type eltType = defaultEltType) {
            super.init(eltType);
            init this;
            this.moduleName = "Flatten";
        }

        override proc forward(input: dynamicTensor(eltType)): dynamicTensor(eltType) do
            return input.flatten();

        override proc attributes(): moduleAttributes do
            return new moduleAttributes("Flatten",moduleName);
    }

    class Parameter : Module(?) {
        var data: dynamicTensor(eltType);

        proc init(data: dynamicTensor(eltType)) {
            super.init(eltType);
            this.data = data;
            init this;
            this.moduleName = "Parameter";
        }

        proc init(data: staticTensor(?rank,?eltType)) do
            this.init(data.eraseRank());
        
        proc init(data: ndarray(?rank,?eltType)) do
            this.init(new staticTensor(data));
        

        proc forward(input: dynamicTensor(eltType)) {
            compilerWarning("Should not be calling forward on a Parameter module");
            return this.data;
        }

        proc attributes(): moduleAttributes do
            return new moduleAttributes(
                "Parameter",
                moduleName,
                ("data","<tensor>")
            );
    }

    class Linear : Module(?) {
        var weights: shared Parameter(eltType);
        var bias: shared Parameter(eltType);
        var inFeatures: int;
        var outFeatures: int;

        proc init(weights: dynamicTensor(?eltType), bias: dynamicTensor(eltType)) {
            super.init(eltType);
            this.weights = new shared Parameter(weights);
            this.bias = new shared Parameter(bias);
            if !weights.checkRank(2) then
                util.err("Weights tensor must have rank 2");
            if !bias.checkRank(1) then
                util.err("Bias tensor must have rank 1");
            
            const weightsShape = weights.forceRank(2).shape;
            const (inFeatures,outFeatures) = weightsShape;
            if outFeatures != bias.forceRank(1).shape[0] then
                util.err("Weights output dimension must match bias input dimension");

            init this;
            this.moduleName = "Linear"
        }

        proc init(type eltType = defaultEltType, inFeatures: int, outFeatures: int) {
            super.init(eltType);
            this.weights = 
        }

        proc forward(input: dynamicTensor(eltType)): dynamicTensor(eltType) do
            return input.matmul(this.weights.data) + this.bias.data;
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