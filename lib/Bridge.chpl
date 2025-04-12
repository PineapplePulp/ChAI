module Bridge {

    import Utilities as util;
    use Utilities.Standard;
    use Allocators;


    extern record bridge_tensor_t {
        var data: c_ptr(real(32));
        var sizes: c_ptr(int(32));
        var dim: int(32);
    }

    proc tensorHandle(type eltType) type {
        if eltType == real(32) then
            return bridge_tensor_t;
        else {
            compilerWarning("BridgeTensorHandle: Unsupported type");
            return bridge_tensor_t;
        }
    }


    extern proc convolve2d(
        in input: bridge_tensor_t, 
        in kernel: bridge_tensor_t, 
        in bias: bridge_tensor_t, 
        in stride: int(32), 
        in padding: int(32)): bridge_tensor_t;

    extern proc unsafe(const ref arr: [] real(32)): c_ptr(real(32));


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

    proc bridgeTensorToArray(param rank: int, package: bridge_tensor_t): [] real(32) {
        const shape = bridgeTensorShape(rank, package);
        const dom = util.domainFromShape((...shape));
        var result: [dom] real(32);
        forall (i,idx) in dom.everyZip() do
            result[idx] = package.data[i];
        deallocate(package.data);
        deallocate(package.sizes);
        return result;
    }


    proc bridgeTensorToExistingArray(ref existing: [] real(32), package: bridge_tensor_t) {
        const shape = bridgeTensorShape(existing.rank, package);
        if existing.shape != shape then
            util.err("BridgeTensorToExistingArray: Shape mismatch");
        const dom = existing.domain;
        forall (i,idx) in dom.everyZip() do
            existing[idx] = package.data[i];
        deallocate(package.data);
        deallocate(package.sizes);
    }

    proc createBridgeTensor(const ref data: [] real(32)): bridge_tensor_t {
        var result: bridge_tensor_t;
        result.data = c_ptrToConst(data) : c_ptr(real(32));
        result.sizes = allocate(int(32),data.rank);
        const sizeArr = getSizeArray(data);
        for i in 0..<data.rank do
            result.sizes[i] = sizeArr[i];

        result.dim = data.rank;
        return result;
    }


}