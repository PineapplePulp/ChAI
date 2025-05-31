module Bridge {
    // require "bridge.h";
    // require "-ltorch";
    require "-ltorch", "-ltorch_cpu", "-lc10", "-ltorch_global_deps";
    require "bridge.h", "-lbridge_objs";

    import Utilities as util;
    use Utilities.Standard;
    use Allocators;

    use CTypes;

    extern record bridge_tensor_t {
        var data: c_ptr(real(32));
        var sizes: c_ptr(uint(32));
        var dim: uint(32);
        var created_by_c: bool;
        var was_freed: bool;
    }

    extern "free_bridge_tensor" proc freeBridgeTensorHandle(
        in tensor: bridge_tensor_t): void;

    extern record bridge_pt_model_t {
        var pt_module: c_ptr(void);
    }
    extern record test_struct_t {
        var field: c_ptr(int(32));
    }

    extern proc hello_world(): void;

    extern record nil_scalar_tensor_t {
        var scalar: real(32);
        var tensor: bridge_tensor_t;
        var is_nil: bool;
        var is_scalar: bool;
        var is_tensor: bool;
    }

    proc tensorHandle(type eltType) type {
        if eltType == real(32) then
            return bridge_tensor_t;
        else {
            compilerWarning("BridgeTensorHandle: Unsupported type");
            return bridge_tensor_t;
        }
    }

    proc torchModuleHandle type do return bridge_pt_model_t;

    extern proc unsafe(const ref arr: [] real(32)): c_ptr(real(32));

    // extern proc load_tensor_from_file(file_path: c_ptrConst(u_char)): bridge_tensor_t; // Working
    extern proc load_tensor_from_file(const file_path: c_ptr(uint(8))): bridge_tensor_t;



    type char_t = uint(8);
    type string_t = c_ptrConst(uint(8));
    extern proc load_tensor_dict_from_file(
        file_path: string_t,
        tensor_key: string_t): bridge_tensor_t;

    extern proc load_run_model(
        model_path: string_t,
        in input: bridge_tensor_t): bridge_tensor_t;

    extern "load_model" proc loadModelC(model_path: string_t): bridge_pt_model_t;
    proc loadModel(modelPath: string): torchModuleHandle {
        const model_path: c_ptr(uint(8)) = c_ptrToConst(modelPath) : c_ptr(uint(8));
        return loadModelC(model_path);
    }

    extern "model_forward" proc modelForward(
        in model: bridge_pt_model_t,
        in input: bridge_tensor_t): bridge_tensor_t;

    extern "accelerator_available" 
        proc acceleratorAvailable(): bool;

    extern "debug_cpu_only_mode" proc debugCpuOnlyMode(mode: bool): void;

    extern proc convolve2d(
        in input: bridge_tensor_t, 
        in kernel: bridge_tensor_t, 
        in bias: bridge_tensor_t, 
        in stride: int(32), 
        in padding: int(32)): bridge_tensor_t;

    extern proc conv2d(
        in input: bridge_tensor_t, 
        in kernel: bridge_tensor_t, 
        in bias: bridge_tensor_t, 
        in stride: int(32), 
        in padding: int(32)): bridge_tensor_t;

    extern proc matmul(in a: bridge_tensor_t, in b: bridge_tensor_t): bridge_tensor_t;

    extern "max_pool2d" proc maxPool2d(
        in input: bridge_tensor_t, 
        in kernel_size: int(32), 
        in stride: int(32),
        in padding: int(32),
        in dilation: int(32)): bridge_tensor_t;

    extern proc resize(
        in input: bridge_tensor_t, 
        in height: int(32), 
        in width: int(32)): bridge_tensor_t;

    extern "imagenet_normalize" proc imageNetNormalize(
        in input: bridge_tensor_t): bridge_tensor_t;

    extern "add_two_arrays" proc addTwoArrays(
        in a: bridge_tensor_t, 
        in b: bridge_tensor_t): bridge_tensor_t;

    extern "split_loop" proc splitLoop(idx: int(64), n: int(64)): void;

    extern "split_loop_filler" proc splitLoopFiller(n: int(64),ret: c_ptr(int(64))): void;

    extern "show_webcam" proc showWebcam(): void;

    // extern "capture_webcam_bridge" proc captureWebcam(
    //     in cam_index: int(32)): bridge_tensor_t;


    proc getSizeArray(const ref arr: [] ?eltType): [] int(32) {
        var sizes: [0..<arr.rank] int(32);
        for i in 0..<arr.rank do
            sizes[i] = arr.dim(i).size : int(32);
        return sizes;
    }

    proc bridgeTensorShape(param dim: int, result: bridge_tensor_t): dim*int {
        var shape: dim*int;
        for i in 0..<dim do
            shape[i] = result.sizes[i] : int;
        return shape;
    }

    proc bridgeTensorToArray(param rank: int,package: bridge_tensor_t): [] real(32) {
        if package.was_freed then
            util.err("BridgeTensorToArray: Tensor has already been freed");
        const shape = bridgeTensorShape(rank, package);
        const dom = util.domainFromShape((...shape));
        var result: [dom] real(32);
        forall (i,idx) in dom.everyZip() do
            result[idx] = package.data[i];
        // This may leak! Alternative is segault on linux. :(
        // if package.created_by_c {
        //     deallocate(package.data);
        //     deallocate(package.sizes);
        // }
        freeBridgeTensor(package);
        return result;
    }


    proc bridgeTensorToExistingArray(ref existing: [] real(32),package: bridge_tensor_t) {
        if package.was_freed then
            util.err("BridgeTensorToArray: Tensor has already been freed");
        const shape = bridgeTensorShape(existing.rank, package);
        if existing.shape != shape then
            util.err("BridgeTensorToExistingArray: Shape mismatch");
        const dom = existing.domain;
        forall (i,idx) in dom.everyZip() do
            existing[idx] = package.data[i];
        freeBridgeTensor(package);
        // if package.created_by_c {
        //     deallocate(package.data);
        //     deallocate(package.sizes);
        // }
    }

    proc freeBridgeTensor(handle: bridge_tensor_t) {
        if handle.created_by_c && !handle.was_freed {
            freeBridgeTensorHandle(handle);
            // handle.was_freed = true;
        }
    }

    proc createBridgeTensor(const ref data: [] real(32)): bridge_tensor_t {
        var result: bridge_tensor_t;
        result.data = c_ptrToConst(data) : c_ptr(real(32));
        result.sizes = allocate(uint(32),data.rank);
        result.created_by_c = false;
        const sizeArr = getSizeArray(data);
        for i in 0..<data.rank do
            result.sizes[i] = sizeArr[i];

        result.dim = data.rank;
        return result;
    }

    extern proc gelu(in input: bridge_tensor_t): bridge_tensor_t;

    extern proc logsigmoid(in input: bridge_tensor_t): bridge_tensor_t;

    extern proc mish(in input: bridge_tensor_t): bridge_tensor_t;

    extern proc relu(in input: bridge_tensor_t): bridge_tensor_t;

    extern proc relu6(in input: bridge_tensor_t): bridge_tensor_t;

    extern proc selu(in input: bridge_tensor_t): bridge_tensor_t;

    extern proc silu(in input: bridge_tensor_t): bridge_tensor_t;

    extern proc softsign(in input: bridge_tensor_t): bridge_tensor_t;

    extern proc tanhshrink(in input: bridge_tensor_t): bridge_tensor_t;

    extern proc rrelu(
        in input: bridge_tensor_t,
        in lower: real(32),
        in upper: real(32),
        in training: bool
    ) : bridge_tensor_t;

    extern proc hardshrink(
        in input: bridge_tensor_t,
        in alpha: real(32)
    ) : bridge_tensor_t;

    extern proc hardtanh(
        in input: bridge_tensor_t,
        in minVal : real(32),
        in maxVal : real(32)
    ) : bridge_tensor_t;

    extern proc elu(
        in input: bridge_tensor_t,
        in alpha: real(32)
    ) : bridge_tensor_t;

    extern proc softplus(
        in input : bridge_tensor_t,
        in beta : real(32),
        in threshold : real(32)
    ) : bridge_tensor_t;

    extern proc threshold(
        in input : bridge_tensor_t,
        in threshold : real(32),
        in value : real(32)
    ) : bridge_tensor_t;

    extern proc celu(
        in input : bridge_tensor_t,
        in alpha : real(32)
    ) : bridge_tensor_t;

    extern "leaky_relu" proc leakyRelu(
        in input : bridge_tensor_t,
        in negativeSlope : real(32)
    ) : bridge_tensor_t;

    extern proc softshrink(
        in input : bridge_tensor_t,
        in alpha : real(32)
    ) : bridge_tensor_t;

    extern proc softmax(
        in input : bridge_tensor_t,
        param dim : int(64)
    ) : bridge_tensor_t;

    extern proc softmin(
        in input : bridge_tensor_t,
        param dim : int(64)
    ) : bridge_tensor_t;

    extern proc dropout(
        in input : bridge_tensor_t,
        in p : real,
        in training : bool
    ) : bridge_tensor_t;

    extern "alpha_dropout" proc alphaDropout(
        in input : bridge_tensor_t,
        in p : real,
        in training : bool
    ) : bridge_tensor_t;

    extern "feature_alpha_dropout" proc featureAlphaDropout(
        in input : bridge_tensor_t,
        in p : real,
        in training : bool
    ) : bridge_tensor_t;

    extern proc dropout2d(
        in input : bridge_tensor_t,
        in p : real,
        in training : bool
    ) : bridge_tensor_t;

    extern proc dropout3d(
        in input : bridge_tensor_t,
        in p : real,
        in training : bool
    ) : bridge_tensor_t;
    
    proc createBridgeTensorWithShape(const ref data: [] real(32),shape: ?rank*int): bridge_tensor_t {
        var result: bridge_tensor_t;
        result.data = c_ptrToConst(data) : c_ptr(real(32));
        result.sizes = allocate(int(32),rank);
        result.created_by_c = false;
        for i in 0..<rank do
            result.sizes[i] = shape(i) : int(32);
        result.dim = rank;
        return result;
    }

    class BridgeTensor {
        var handle: bridge_tensor_t;

        proc init(bt: bridge_tensor_t) do
            this.handle = bt;
        
        proc init(const ref data: [] real(32)) {
            var bt: bridge_tensor_t;
            bt.data = c_ptrToConst(data) : c_ptr(real(32));
            bt.sizes = allocate(uint(32),data.rank);
            bt.created_by_c = false;
            bt.was_freed = false;
            const sizeArr = getSizeArray(data);
            for i in 0..<data.rank do
                bt.sizes[i] = bt[i];
            bt.dim = data.rank;
            this.init(bt);
        }

        proc deinit() {
            if this.handle.was_freed then
                util.err("BridgeTensor: Tensor has already been freed");
            if this.handle.created_by_c && !this.handle.was_freed {
                freeBridgeTensorHandle(this.handle);
                this.handle.was_freed = true;
            }
        }
        
        proc getShape(param dim: int): dim*int {
            var shape: dim*int;
            for i in 0..<dim do
                shape[i] = handle.sizes[i] : int;
            return shape;
        }
    }
}