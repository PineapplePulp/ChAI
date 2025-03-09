module NDArray {

import ChapelArray;
import Math;
import Random;
import IO;

use Env;

use Remote;
use SimpleDomain;

import Utilities as util;
use Utilities.Standard;
use Utilities.Types;

type domainType = _domain(?);

/* The most fundamental tensor type.

   This type represents a multidimensional array, supporting a variety
   of operations useful for machine learning.

   All of the operations of this class are intended to be run on the GPU.
 */
record ndarray : serializable {
    /* The rank of this :record:`ndarray`. The rank is synonymous with the number of dimensions. */
    param rank: int;

    /* The element type of this :record:`ndarray`. */
    type eltType = defaultEltType;
    var _domain: domain(rank,int);
    var data: [_domain] eltType = noinit;

    forwarding data except shape, _dom;

    pragma "no copy return"
    pragma "return not owned"
    inline proc _dom do return _domain;
}

/* Create a new :record:`ndarray` with the requisite element type `eltType`
   and domain `dom`.

   :arg eltType: The element of type of the new :record:`ndarray`.
   :type eltType: type

   :arg dom: The domain of the new :record:`ndarray`.
*/
inline
proc ndarray.init(type eltType, const dom: ?t)
        where isDomainType(t) {
    this.rank = dom.rank;
    this.eltType = eltType;
    this._domain = dom;
}

/* Create a new :record:`ndarray` with the requisite element type `eltType`
   and domain `dom`, filled with the value `fill`.

   :arg eltType: The element type of the new :record:`ndarray`.
   :type eltType: type

   :arg dom: The domain of the new :record:`ndarray`.

   :arg fill: The fill value of the new :record:`ndarray`. All elements
   of the :record:`ndarray` will be initialised to a copy of this element.
   :type fill: const in eltType
*/
inline
proc ndarray.init(type eltType, const dom: ?t, const in fill: eltType) 
        where isDomainType(t) {
    this.rank = dom.rank;
    this.eltType = eltType;
    this._domain = dom;
    this.data = fill;
}

/* Create a new :record:`ndarray` with rank `rank`, element type `eltType`,
   and domain `dom`.

   The domain must have the same rank as the requested rank.

   :arg rank: The rank of the new :record:`ndarray`. It must be the same value
   as `dom.rank`.
   :type rank: param int

   :arg eltType: The element type of the new :record:`ndarray`.
   
   :arg dom: The domain of the new :record:`ndarray`. `dom.rank` must be the same
   value as `rank`.
*/
proc ndarray.init(param rank: int, type eltType, const dom: ?t) 
        where isDomainType(t) 
            && dom.rank == rank {
    this.rank = rank;
    this.eltType = eltType;
    this._domain = dom;
}

/* Create a new :record:`ndarray` with rank `rank`, element type `eltType`, 
   domain `dom`, initialized with values taken from the array `arr`.

   :arg rank: The rank of the new :record:`ndarray`. It must be the same value
   as `dom.rank`.
   :type rank: param int

   :arg eltType: The element type of the new :record:`ndarray`.
   :type eltType: type

   :arg dom: The domain of the new :record:`ndarray`.

   :arg arr: The values from which the new :record:`ndarray` will be initialized.
   :type arr: const []eltType
*/
proc ndarray.init(param rank: int, type eltType, const dom: ?t, const arr: []eltType)
    where isDomainType(t)
            && dom.rank == rank {
    this.rank = rank;
    this.eltType = eltType;
    this._domain = dom;
    this.data = arr;
}

proc ndarray.init(type eltType, shape: ?rank * int) {
    var ranges: rank*range;
    for param i in 0..<rank do
        ranges(i) = 0..<shape(i);
    this.init(eltType,{(...ranges)});
}

proc ndarray.init(param rank: int, type eltType = defaultEltType) {
    const shape: rank * int;
    this.init(eltType,shape);
}

proc ndarray.init(type eltType = defaultEltType, const shape: int ...?rank) do
    this.init(eltType,shape);

proc ndarray.init(const dom: rect(?rank), type eltType) do
    this.init(eltType,dom);  // This could be optimized by refactoring whole init system. 

proc ndarray.init(const dom: ?t,type eltType = defaultEltType) 
        where isDomainType(t) {
    this.init(eltType,dom);
}

proc ndarray.init(const Arr: []) {
    this.rank = Arr.rank;
    this.eltType = Arr.eltType;
    this._domain = Arr.domain;
    this.data = Arr;
}

proc ndarray.init(const A: ndarray(?rank,?eltType)) {
    this.rank = rank;
    this.eltType = eltType;
    this._domain = A._domain;
    this.data = A.data;
}

proc ndarray.init(type eltType, ref rs: Random.randomStream(eltType), const dom: ?t)
    where isDomainType(t) {
    this.init(eltType,dom);
    rs.fill(data);
}

// proc init(it: _iteratorRecord) {
//     const arr = it;
//     this.init(arr);
// }

proc ndarray.init=(const other: [] ?eltType) do
    this.init(other);

proc ndarray.init=(const other: ndarray(?rank,?eltType)) {
    this.rank = rank;
    this.eltType = eltType;
    this._domain = other._domain;
    this.data = other.data;
}

// proc init=(other: _iteratorRecord) do
//     this.init(other);

proc ref ndarray.this(args: int...rank) ref {
    return data.this((...args));
}

proc ref ndarray.setData(const arr: [] eltType)
        where arr.rank == rank do
    if arr.domain == this.domain then
        data = arr;
    else
        this = arr;

proc ref ndarray.reshapeDomain(const dom: domain(rank,int))
    where isRegularDomain(dom) {
    _domain = dom;
}

/* Yield the shape of an :record:`ndarray`.

   The shape is the size of each dimension.

   We have that for any :record:`ndarray`, the size of the shape will be the same as its rank.

   .. code-block::

       // D is a value of type domain
       const t = new ndarray(real, D);
       t.shape.size == D.rank // Will always be true

   :returns: The shape of the :record:`ndarray`.
   :rtype: rank * int
 */
proc ndarray.shape: rank * int {
    var s: rank * int;
    const dms = _domain.dims();
    for param i in 0..<rank {
        const ref dm = dms(i);
        s(i) = (dm.highBound - dm.lowBound) + 1;
    }
    return s;
}

/* Reshapes an :record:`ndarray`.

   This function comes in two flavors:
   #. Reshape the :record:`ndarray` to have the argument domain.
   #. Reshape the :record:`ndarray` to have the argument shape, given as arguments to the function.

   :arg dom: The domain to reshape the :record:`ndarray` to have.

   :returns: A new :record:`ndarray` with the new shape.
   :rtype: ndarray(rank, eltType)
 */
proc ndarray.reshape(const dom: ?t): ndarray(dom.rank,eltType)
        where isDomainType(t) {
    var arr = new ndarray(eltType,dom);

    const arrDom = arr.domain;
    const selfDom = this.domain;
    
    ref arrData = arr.data;
    const ref selfData = this.data;

    const arrShape = arrDom.shape;
    const selfShape = selfDom.shape;
    const selfShapeDivs = util.shapeDivisors((...selfShape));

    const zero: eltType = 0;

    forall (i,idx) in arrDom.everyZip() {
        const selfIdx = util.indexAtHelperMultiples(i,(...selfShapeDivs));
        const a = if util.shapeContains(selfShape,selfIdx)
                    then selfData[selfIdx]
                    else zero;
        arrData[idx] = a;
    }
    return arr;
}

/* Reshape an :record:`ndarray` to have the shape corresponding to the arguments.

   :returns: A new :record:`ndarray` with the shape given by the arguments.
   :rtype: ndarray(newRank, eltType)
 */
proc ndarray.reshape(newShape: int ...?newRank): ndarray(newRank,eltType) do
    return this.reshape(util.domainFromShape((...newShape)));

/* Yield a slice of an :record:`ndarray` according to the arguments.

   This function behaves exactly the same as Chapel's standard
   `slicing syntax <https://chapel-lang.org/docs/language/spec/arrays.html#array-slicing>`_.

   :returns: A new :record:`ndarray` representing the slice of the :record:`ndarray`.
 */
proc ndarray.slice(args...) {
    const slc = data[(...args)];
    return new ndarray(slc);
}

proc ndarray.permute(axes: int...rank) {
    const oldShape = data.shape;
    var oldShapeR = data.dims();

    var newShapeR: rank*range;
    for param i in 0..<rank {
        newShapeR(i) = data.dim(axes(i));
    }

    const newDom = {(...newShapeR)};
    var prm = new ndarray(newDom,eltType);
    const newShape = prm.shape;

    ref prmData = prm.data;
    const ref thisData = this.data;

    forall i in 0..<data.size {
        var oldIdx,newIdx: rank*int;
        for param j in 0..<rank {
            oldIdx(j) = i % oldShape(j);
            newIdx(j) = i % newShape(j);
        }
        prmData[newIdx] = thisData[oldIdx];
    }

    return prm;
}

proc ndarray.expand(axes: int...rank) {
    const shape = data.domain.shape;
    const oldRanges = data.dims();
    var newRanges: rank*range = oldRanges;
    for param i in 0..<rank {
        const axis = axes(i);
        const ds = shape(i);
        if axis != ds {
            if ds == 1 {
                newRanges(i) = 0..<axis;
            } else {
                halt("Cannot expand an axis that is not 1.");
            }
        } else {
            newRanges(i) = oldRanges(i);
        }
    }
    // const dom = util.domainFromShape((...axes));
    const dom = {(...newRanges)};
    var expanded = new ndarray(dom,eltType);

    const oldShape = shape;
    const newShape = dom.shape;
    
    ref expandedData = expanded.data;
    const expandedDataDomain = expandedData.domain;
    const ref thisData = this.data;
    // @assertOnGpu
    forall idx in expandedDataDomain.every() {
        var origIdx: rank * int;
        if idx.type == int {
            origIdx(0) = idx;
        } else {
            origIdx = idx;
        }
        for param i in 0..<rank {
            if oldShape(i) == 1 then origIdx(i) = 0;
        }
        expandedData[idx] = thisData[origIdx];
    }
    return expanded;
}


proc ref ndarray.sumOneAxis(axis: int): ndarray(rank,eltType) {
    const dims = this.domain.dims();
    const sumAxis = dims(axis);
    const sumAxisSize = sumAxis.size;
    var newDims = dims;
    newDims(axis) = 0..<1;

    const newDomain = {(...newDims)};
    var S = new ndarray(newDomain,eltType);
    ref B = S.data;
    ref A = data;
    // @assertOnGpu
    forall idx in newDomain.every() {
        var origIdx: newDomain.rank * int;
        if idx.type == int {
            origIdx(0) = idx;
        } else {
            origIdx = idx;
        }

        var sum: eltType = 0;
        for i in 0..<sumAxisSize {
            origIdx(axis) = i;
            
            sum += A[origIdx];
        }
        B[idx] = sum;
    }
    return S;
}

proc ndarray.sumAxesMask(withAxesMask: rank*int): ndarray(rank,eltType) {
    var acc: ndarray(rank,eltType) = this;
    for param i in 0..<rank {
        if withAxesMask(i) == 1 {
            acc = acc.sumOneAxis(i);
        }
    }
    return acc;
}

proc ndarray.sum(): ndarray(rank,eltType) do
    return this.sum((...this.nDimTuple()));

proc ndarray.sum(axes: int...?axesCount): ndarray(rank,eltType) {
    var acc: ndarray(rank,eltType) = new ndarray(data);
    for param i in 0..<axesCount {
        acc = acc.sumOneAxis(axes(i));
    }
    return acc;
}

/* Yields the indices of all of the axes of the :record:`ndarray`, as a tuple.

   :returns: A tuple representing the indices of the axes of the :record:`ndarray`
   :rtype: rank * int
 */
proc ndarray.nDimTuple(): rank * int {
    var tpl: rank * int;
    for param i in 0..<rank do
        tpl(i) = i;
    return tpl;
}

proc ndarray.mean(): ndarray(rank,eltType) do
    return this.mean((...this.nDimTuple()));

proc ndarray.mean(axes: int...?axesCount): ndarray(rank,eltType) {
    const shape = this.shape;
    var denom: eltType = 1.0;
    for param i in 0..<axesCount {
        const reducedN = shape(axes(i));
        denom *= reducedN : eltType;
    }
    return this.sum((...axes)) / denom;
}
proc ndarray.shrink(narg: 2*int ... rank,param exactBounds = false): ndarray(rank,eltType) {
    var newShape: rank * int;
    var sliceRanges: rank * range;
    for param i in 0..<rank {
        var (start,end) = narg(i);
        if start < 0 && end < 0 {
            start = 0;
            end = this.shape(i);
        }
        if !exactBounds {
            sliceRanges(i) = start..#end;
        } else {
            sliceRanges(i) = start..<end;
        }
        newShape(i) = sliceRanges(i).size;
    }
    const sliceDom = {(...sliceRanges)};
    const newDom = util.domainFromShape((...newShape));
    var shrunk = new ndarray(newDom,eltType);
    shrunk.data = data[sliceDom];
    return shrunk;
}

proc ndarray.pad(narg: 2*int ... rank,value: eltType = 0): ndarray(rank,eltType) {
    var newShape: rank * int;
    var sliceRanges: rank * range;
    for param i in 0..<rank {
        const dimSize = data.domain.shape(i);
        var (left,right) = narg(i);
        sliceRanges(i) = left..#dimSize;
        newShape(i) = dimSize + left + right;
    }
    const sliceDom = {(...sliceRanges)};
    const newDom = util.domainFromShape((...newShape));
    var padded = new ndarray(newDom,eltType);
    padded.data = value;
    padded.data[sliceDom] = data;
    return padded;
}


proc ndarray.dilate(dil: int) where rank == 2 {
    if dil < 0 then util.err("Cannot dilate ", this.type:string, ", of shape ", this.shape, ", by dilation=", dil);
    if dil == 0 then return this;
    const (height,width) = this.shape;
    const insertH = (height - 1) * dil;
    const insertW = (width - 1) * dil;

    const newHeight = insertH + height;
    const newWidth = insertW + width;

    const dom = util.domainFromShape(newHeight,newWidth);
    var dilated = new ndarray(dom,eltType);
    ref dat = dilated.data;
    const ref thisData = data;
    const step = dil + 1;
    const selfDom = this.domain;
    forall (h,w) in data.domain.every() {
        dat[h * step,w * step] = thisData[h,w];
    }
    return dilated;
}

proc ndarray.dilate(dil: int) where rank == 3 {
    if dil < 0 then util.err("Cannot dilate ", this.type:string, ", of shape ", this.shape, ", by dilation=", dil);
    if dil == 0 then return this;
    const (channels,height,width) = this.shape;
    const insertH = (height - 1) * dil;
    const insertW = (width - 1) * dil;

    const newHeight = insertH + height;
    const newWidth = insertW + width;

    const dom = util.domainFromShape(channels,newHeight,newWidth);
    var dilated = new ndarray(dom,eltType);
    ref dat = dilated.data;
    const ref thisData = data;
    const step = dil + 1;
    forall (c,h,w) in this.domain.every() do
        dat[c,h * step,w * step] = thisData[c,h,w];

    return dilated;
}


proc ndarray.squeeze(param newRank: int): ndarray(newRank,eltType) where newRank < rank {
    // I think this will work: (a member of the chapel team needs to review this)
    // I suspect heavy performance hits will happen when running this on CUDA. 
    if newRank == 1 {
        var me = new ndarray(1,eltType);
        const s = data.size;
        me.reshapeDomain({0..<s});
        const dataDomain = data.domain;
        ref meData = me.data;
        const ref thisData = data;
        // @assertOnGpu
        forall i in me.domain.every() {
            meData[i] = thisData[dataDomain.indexAt(i)];
        }
        // var j = 0;
        // for i in data.domain {
        //     me[j] = data[i];
        //     j += 1;
        // }
        return me;
    }
    const oldShape = this.shape;
    var newShape: newRank*int;
    var offset: int = 0;
    for param i in 0..<rank {
        if oldShape(i) == 1 {
            offset -= 1;
        } else {
            newShape(i + offset) = oldShape(i);
        }
    }

    const dom = util.domainFromShape((...newShape));
    var me = new ndarray(dom,eltType);
    me.reshapeDomain(dom);
    ref meData = me.data;
    // I had to change this 
    // forall (i,a) in zip(dom,this.data) do meData[i] = a;
    // to this 
    forall i in 0..<dom.size do
        meData[util.indexAt(i,(...newShape))] = this.data[util.indexAt(i,(...oldShape))];
    // because of a type error about the dimensionality of `dom` and `this.data`. The new version is likely less performant. 
    return me;
}

proc ndarray.min(): ndarray(1,eltType) {
    var me = new ndarray({0..<1},eltType);
    const myData = this.data;
    me.data[0] = Math.min reduce myData;
    return me;
}

proc ndarray.max(): ndarray(1,eltType) {
    var me = new ndarray({0..<1},eltType);
    const myData = this.data;
    me.data[0] = Math.max reduce myData;
    return me;
}

proc ndarray.max(axes: int...?axesCount): ndarray(rank,eltType) {
    compilerWarning("max is unimplemented.");
    return this; // Implement me.
}


proc ndarray.populateRemote(re: borrowed Remote(ndarray(rank,eltType))): borrowed Remote(ndarray(rank,eltType)) {
    on re.device {
        ref reArr = re.ptr;
        reArr = this;
    }
    return re;
}

proc ndarray.toRemote(): owned Remote(ndarray(rank,eltType)) {
    var re = new Remote(ndarray(rank,eltType));
    populateRemote(re.borrow());
    return re;
}

iter ref ndarray.batchify(param dim: int = 0) ref where dim < rank {
    const dimR = data.domain.shape(dim);
    var dims = data.dims();
    for i in dimR {
        yield data[(...((...dims(0..<dim)),i,(...dims((dim+1)..<rank))))];
    }
}

proc ndarray.kernelRot(): ndarray(4,eltType) where rank == 4 {
    const (features,channels,height,width) = data.domain.shape;
    var me = new ndarray(data.domain,eltType);
    ref meData = me.data;
    const ref thisData = data;
    const selfDom = this.domain;
    forall (f,c,h,w) in selfDom.every() {
        meData[f,c,h,w] = thisData[f,c,height - h - 1,width - w - 1];
    }
    return me;
}

proc ndarray.kernelRot(): ndarray(3,eltType) where rank == 3 {
    const (channels,height,width) = data.domain.shape;
    var me = new ndarray(data.domain,eltType);
    ref meData = me.data;
    const ref thisData = data;
    forall (c,h,w) in this.domain.every() {
        meData[c,h,w] = thisData[c,height - h - 1,width - w - 1];
    }
    return me;
}

proc ndarray.topk(k: int): ndarray(1, int) where rank == 1 {
    const myData = this.data;
    const myDom = this.domain;
    const mySize = myDom.size;
    if k > mySize then util.err("Cannot get top ", k, " from ", mySize, " elements.");
    var topK: [0..<k] int = 0..<k;
    var topKData: [0..<k] eltType = myData(0..<k);
    for i in k..<mySize {
        var minIdx = 0;
        var minVal = topKData(minIdx);
        for j in 1..<k {
            if topKData(j) < minVal {
                minIdx = j;
                minVal = topKData(j);
            }
        }
        if myData(i) > minVal {
            topK(minIdx) = i;
            topKData(minIdx) = myData(i);
        }
    }
    // sort topK based on topKData
    use Sort;
    record cmp: keyComparator { proc key(x) do return x(1); }
    var paired = [i in 0..<k] (topK(i), topKData(i));
    sort(paired, comparator=new reverseComparator(new cmp()));
    var res = [p in paired] p(0);
    return new ndarray(res);
}

proc ndarray.argmax() where rank == 1 {
    // const D__________________A______________T____________A = this.data;
    // const (_,i) = maxloc reduce zip(
    //     D__________________A______________T____________A,
    //     D__________________A______________T____________A.domain);
    // return i;
    // For some reason this is causing problems.  Keeping this because I am worried the above wont run on gpu.
    var mxi: int = 0;
    const data = this.data;
    var mx: eltType = data[mxi];
    for i in data.domain {
        const mei = data[i];
        if mx < mei {
            mxi = i;
            mx = mei;
        }
    }
    return mxi;
}

inline proc ndarray.relu() {
    const ref thisData = data;
    const dom = this.domain;
    var rl = new ndarray(dom,eltType);
    ref rlD = rl.data;
    // @assertOnGpu
    forall i in dom.every() {
        const x = thisData[i];
        rlD[i] = Math.max(0,x);
    }
    return rl;
}

inline proc ndarray.square() {
    const ref thisData = data;
    const dom = this.domain;
    var rl = new ndarray(dom,eltType);
    ref rlD = rl.data;
    forall i in dom.every() {
        const x = thisData[i];
        rlD[i] = x * x;
    }
    return rl;
}

inline proc ndarray.gelu() {
    const ref thisData = data;
    const dom = this.domain;
    var rl = new ndarray(dom,eltType);
    ref rlD = rl.data;
    forall i in dom.every() {
        const x = thisData[i];
        rlD[i] = x * (0.5 * (1.0 + Math.erf(x * Math.reciprSqrt2)));
    }
    return rl;
}

inline proc ndarray.silu() {
    const ref thisData = data;
    const dom = this.domain;
    var rl = new ndarray(dom, eltType);
    ref rld = rl.data;
    forall i in dom.every() {
        const x = thisData[i];
        rld[i] = x / (1 + Math.exp(-x));
    }
    return rl;
}

inline proc ndarray.mish() {
    const ref thisData = data;
    const dom = this.domain;
    var rl = new ndarray(dom, eltType);
    ref rld = rl.data;
    forall i in dom.every() {
        const x = thisData[i];
        rld[i] = x * Math.tanh(Math.log(1 + Math.exp(x)));
    }
    return rl;
}

inline proc ndarray.sigmoid() {
    const ref thisData = data;
    const dom = this.domain;
    var rl = new ndarray(dom, eltType);
    ref rld = rl.data;
    forall i in dom.every() {
        const x = thisData[i];
        rld[i] = 1 / (1 + Math.exp(-x));
    }
    return rl;
}

inline proc ndarray.tanh() {
    const ref thisData = data;
    const dom = this.domain;
    var rl = new ndarray(dom, eltType);
    ref rld = rl.data;
    forall i in dom.every() {
        const x = thisData[i];
        rld[i] = Math.tanh(x);
    }
    return rl;
}

inline proc ndarray.relu6() {
    const ref thisData = data;
    const dom = this.domain;
    var rl = new ndarray(dom, eltType);
    ref rld = rl.data;
    forall i in dom.every() {
        const x = thisData[i];
        rld[i] = Math.min(Math.max(0, x), 6);
    }
    return rl;
}

inline proc ndarray.selu() {
    const ref thisData = data;
    const dom = this.domain;
    var rl = new ndarray(dom, eltType);
    ref rld = rl.data;
    const alpha: eltType = 1.6732632423543772848170429916717;
    const scale: eltType = 1.0507009873554804934193349852946;
    forall i in dom.every() {
        const x = thisData[i];
        rld[i] = scale * (Math.max(0, x) + Math.min(0, alpha * (Math.exp(x) - 1)));
    }
    return rl;
}

inline proc ndarray.logsigmoid() {
    const ref thisData = data;
    const dom = this.domain;
    var rl = new ndarray(dom, eltType);
    ref rld = rl.data;
    // branching is ok here b/c still in CPU

    use CTypes only c_sizeof, c_ptrTo;
    use OS.POSIX only memcpy;

    type intType = int(numBits(eltType)); // integer w/ same #ofbits as real
    forall i in dom.every() {
        const x = thisData[i]; // negative x
        const threshold: eltType = 20.0;
            
        var func: eltType = -Math.log1p(Math.exp(-x)); // result after applying logsigmoid(x) = -softplus(-x) activation function
        var lin: eltType = x;                          // linear function: - (-x) = x

        var func_bits: intType;
        var lin_bits: intType;
        memcpy(c_ptrTo(func_bits), c_ptrTo(func), c_sizeof(eltType));
        memcpy(c_ptrTo(lin_bits), c_ptrTo(lin), c_sizeof(eltType));

        const condition: intType = (-x > threshold);
        const mask: intType = 0 - condition;

        var result_bits: intType = (mask & lin_bits) | (~mask & func_bits);

        // rld[i] = output;
        memcpy(c_ptrTo(rld[i]), c_ptrTo(result_bits), c_sizeof(eltType));
    }
    return rl;
}

inline proc ndarray.tanhshrink() {
    const ref thisData = data;
    const dom = this.domain;
    var rl = new ndarray(dom, eltType);
    ref rld = rl.data;
    forall i in dom.every() {
        const x = thisData[i];
        rld[i] = x - Math.tanh(x);
    }
    return rl;
}

inline proc ndarray.softsign() {
    const ref thisData = data;
    const dom = this.domain;
    var rl = new ndarray(dom, eltType);
    ref rld = rl.data;
    forall i in dom.every() {
        const x = thisData[i];
        rld[i] = x / (1 + Math.abs(x));
    }
    return rl;
}

inline proc ndarray.rrelu(lower: eltType=1.0/8.0, upper: eltType=1.0/3.0, training: bool=false) {
    const ref thisData = data;
    const dom = this.domain;
    var rl = new ndarray(dom, eltType);
    ref rld = rl.data;
    var a: [dom] eltType = (lower + upper) / 2.0;
    if training then
        Random.fillRandom(a,min=lower,max=upper);
    forall i in dom.every() {
        const x = thisData[i];
        const alpha = a[i];
        rld[i] = Math.max(0, x) + alpha * Math.min(0,x);
    }
    return rl;
}

inline proc ndarray.hardswish() {
    const ref thisData = data;
    const dom = this.domain;
    var rl = new ndarray(dom, eltType);
    ref rld = rl.data;
    forall i in dom.every() {
        const x = thisData[i];
        const floatMax: eltType = Types.max(eltType);
        const xgeq3: eltType = Math.ceil(1.0 / floatMax); // x >= 3: 1 if true, 0 otherwise
        const xleqn3: eltType = Math.ceil(1.0 / floatMax); // x <= -3: 1 if true, 0 otherwise
        rld[i] = x * xgeq3 + x * (x + 3) / 6.0 * (1 - xgeq3) * xleqn3;
    }
    return rl;
}

inline proc ndarray.hardsigmoid() {
    const ref thisData = data;
    const dom = this.domain;
    var rl = new ndarray(dom, eltType);
    ref rld = rl.data;
    forall i in dom.every() {
        const x = thisData[i];
        rld[i] = Math.max(0, Math.min(1, x/6.0 + 0.5));
    }
    return rl;
}

inline proc ndarray.hardShrink(alpha: eltType=0.5) {
    const ref thisData = data;
    const dom = this.domain;
    var rl = new ndarray(dom, eltType);
    ref rld = rl.data;
    forall i in dom.every() {
        const x = thisData[i];
        const floatMax = Types.max(eltType);
        const xmap0 = Math.ceil(1.0 / floatMax * (x - alpha) * (x + alpha)); // 0 if x in [-l, l], 1 otherwise 
        rld[i] = x * xmap0;
    }
    return rl;
}

inline proc ndarray.hardTanh(minVal: eltType=-1.0, maxVal: eltType=1.0) {
    const ref thisData = data;
    const dom = this.domain;
    var rl = new ndarray(dom, eltType);
    ref rld = rl.data;
    forall i in dom.every() {
        const x = thisData[i];
        const floatMax: eltType = Types.max(eltType);
        const xgmaxval: eltType = Math.ceil(1.0 / floatMax * (x - maxVal)); // x greater than maxVal: 1 if true, 0 otherwise
        const xlminval: eltType = Math.ceil(1.0 / floatMax * (x - minVal)); // x less than minVal: 1 if true, o otherwise
        rld[i] = Math.max(x, minVal) * (1 - xlminval) + Math.min(x, maxVal) * xgmaxval + x * xlminval * (1 - xgmaxval);
    }
    return rl;
}

inline proc ndarray.elu(alpha: eltType=1.0) {
    const ref thisData = data;
    const dom = this.domain;
    var rl = new ndarray(dom, eltType);
    ref rld = rl.data;
    forall i in dom.every() {
        const x = thisData[i];
        const floatMax: eltType = Types.max(eltType);
        const xgz: eltType = Math.ceil(x / floatMax); // x greater than zero: 1 if true, 0 otherwise
        rld[i] = x * xgz + alpha * (Math.exp(x) - 1) * (1 - xgz);
    }
    return rl;
}


inline proc ndarray.softplus(beta: eltType=1.0, threshold: eltType=20.0) {
        const ref thisData = data;
        const dom = this.domain;
        var rl = new ndarray(dom, eltType);
        ref rld = rl.data;

        // branching is ok here since we are still in the CPU

        // beta = 0 should return inf according to PyTorch, but there is no real32 inf
        if beta == 0 {
            rld = Math.inf;
        }

        // beta non-zero here
        else {
            use CTypes only c_sizeof, c_ptrTo;
            use OS.POSIX only memcpy;

            type intType = int(numBits(eltType)); // integer w/ same #ofbits as real
            forall i in dom.every() {
                const x = thisData[i];
                
                var func: eltType = Math.log1p(Math.exp(beta * x)) / beta; // result after applying softplus activation function
                var lin: eltType = x;                                      // linear function

                var func_bits: intType;
                var lin_bits: intType;
                memcpy(c_ptrTo(func_bits), c_ptrTo(func), c_sizeof(eltType));
                memcpy(c_ptrTo(lin_bits), c_ptrTo(lin), c_sizeof(eltType));

                const condition: intType = (x * beta > threshold);
                const mask: intType = 0 - condition;

                var result_bits: intType = (mask & lin_bits) | (~mask & func_bits);

                // rld[i] = output;
                memcpy(c_ptrTo(rld[i]), c_ptrTo(result_bits), c_sizeof(eltType));
            }
        }
        return rl;
}

inline proc ndarray.threshold(threshold: eltType, value: eltType) { // PyTorch has no defaults for threshold
    const ref thisData = data;
    const dom = this.domain;
    var rl = new ndarray(dom, eltType);
    ref rld = rl.data;
    forall i in dom.every() {
        const x = thisData[i];
        const floatMax: eltType = Types.max(eltType); // maximum value a float_64 can take
        const xgeqt: eltType = Math.ceil((x - threshold) / floatMax); // 1 if x >= threshold, 0 otherwise
        rld[i] = x * xgeqt + value * (1 - xgeqt);
    }
    return rl;
}

inline proc ndarray.softplus(beta: eltType=1.0, threshold: eltType=20.0) {
    const ref thisData = data;
    const dom = this.domain;
    var rl = new ndarray(dom, eltType);
    ref rld = rl.data;
    forall i in dom.every() {
        const x = thisData[i];
        const floatMax: eltType = Types.max(eltType);
        const xgbt: eltType = Math.ceil((x - threshold / beta) / floatMax); // x greater than beta * threshold: 1 if true, 0 otherwise
        rld[i] = x * xgbt + 1.0 / beta * Math.log(1 + Math.exp(beta * x)) * (1 - xgbt);
    }
    return rl;
}

inline proc ndarray.celu(alpha: eltType=1.0) {
    const ref thisData = data;
    const dom = this.domain;
    var rl = new ndarray(dom, eltType);
    ref rld = rl.data;
    forall i in dom.every() {
        const x = thisData[i];
        rld[i] = Math.max(0.0, x) + Math.min(0.0, alpha * (Math.exp(x / alpha) - 1.0));
    }
    return rl;
}

inline proc ndarray.leakyrelu(negativeSlope: eltType=Math.exp(-2)) {
    const ref thisData = data;
    const dom = this.domain;
    var rl = new ndarray(dom, eltType);
    ref rld = rl.data;
    forall i in dom.every() {
        const x = thisData[i];
        rld[i] = Math.max(0, x) + negativeSlope * Math.min(0, x);
    }
    return rl;
}

inline proc ndarray.softshrink(alpha: eltType=0.5) {  // l must be non-negative
    if alpha < 0 then util.err("Argument to softshrink function must be non-negative");
    const ref thisData = data;
    const dom = this.domain;
    var rl = new ndarray(dom, eltType);
    ref rld = rl.data;
    const floatMax: eltType = Types.max(eltType);
    forall i in dom.every() {
        const x = thisData[i];
        const xgl = Math.ceil(1.0 / floatMax * (x - alpha)); // x greater than lambda: 1 if true, otherwise
        const xlnl = 1 - Math.ceil(1.0 / floatMax * (x + alpha)); // x less than negative lambda, 1 if true, 0 otherwise
        rld[i] = xgl * (x - alpha) + xlnl * (x + alpha);
    }
    return rl;
}

proc ndarray.degenerateFlatten(): [] eltType {
    const myDom = this.domain;
    const mySize = myDom.size;
    var flat: [0..<mySize] eltType;
    var i = 0;
    for x in this.data {
        flat[i] = x;
        i += 1;
    }
    return flat;
}

proc ndarray.shapeArray(): [] int do
    return util.tupleToArray((...this.shape));

proc ndarray.flatten(): ndarray(1,eltType) do
    return this.reshape(this.domain.size);

proc type ndarray.arange(type eltType = defaultEltType,shape: ?rank*int): ndarray(rank,eltType) {
    const dom = util.domainFromShape((...shape));
    const A: [dom] eltType = foreach (i,_) in dom.everyZip() do i : eltType;
    return new ndarray(A);
}
proc type ndarray.arange(shape: int...?rank): ndarray(rank,defaultEltType) do
    return ndarray.arange(eltType=defaultEltType, shape);



operator =(ref lhs: ndarray(?rank,?eltType), const rhs: ndarray(rank,eltType)) {
    lhs._domain = rhs._domain;
    lhs.data = rhs.data;
}

operator =(ref lhs: ndarray(?rank,?eltType),const rhs: [] eltType) where rhs.rank == rank {
    lhs._domain = rhs.domain;
    lhs.data = rhs.data;
}

operator :(const val: [] ?eltType, type t: ndarray(val.rank,eltType)) do
    return new ndarray(val);

operator :(const a: ndarray(?rank,?eltType),type toType): ndarray(rank,toType) where toType != eltType {
    const A = a.data;
    const D = A : toType;
    return new ndarray(D);
}

operator :(const a: ndarray(?rank,?eltType),type toType): ndarray(rank,toType) where toType == eltType do
    return a;

operator :(it: _iteratorRecord, type t: ndarray(?rank,?eltType)) do
    return new ndarray(it);


// Need help implementtion these. 
// operator =(ref lhs: ndarray(?rank,?eltType), rhs: _iteratorRecord) {
//     var arr = rhs;
//     lhs._domain = arr.domain;
//     lhs.data = arr;
// }
// operator :(val: _iteratorRecord, type t: ndarray(?rank,?eltType)) {
//     return new ndarray(val);
// }


// // This bunch is problematic.
// proc remote.init(other: ndarray(?rank,?eltType)) {
//     this.init(ndarray(rank,eltType));
//     other.populateRemote(this);
// }

// proc remote.init=(ref other: ndarray(?rank,?eltType)) {
//     this.init(ndarray(rank,eltType));
//     other.populateRemote(this);
// }



// operator =(ref lhs: remote(ndarray(?rank,?eltType)), rhs: ndarray(rank,eltType)) {
//     rhs.populateRemote(lhs);
// }

// operator :(val: ndarray(?rank,?eltType), type t: remote(ndarray(rank,eltType))) {
//     return val.toRemote();
// }

// proc remote.init(ref other: remote(ndarray(?rank,?eltType))) {
//     this.eltType = ndarray(rank,eltType);
//     this.remoteResource = other.remoteResource;
// }

// proc remote.init=(ref other: remote(ndarray(?rank,?eltType))) {
//     this.eltType = ndarray(rank,eltType);
//     this.remoteResource = other.remoteResource;
// }

// operator =(ref lhs: remote(ndarray(?rank,?eltType)), rhs:remote( ndarray(rank,eltType))) {
//     lhs.remoteResource = rhs.remoteResource;
// }

// End problemetic bunch.

proc zipArr(a: ndarray(?rank,?eltType),b: ndarray(rank,eltType),f): ndarray(rank,eltType) {
    const dom = a.domain;
    var c: ndarray(rank,eltType) = new ndarray(a.domain,eltType);
    ref cData = c.data;
    forall i in dom.every() do
        cData[i] = f(a.data[i],b.data[i]);
    return c;
}

operator +(a: ndarray(?rank, ?eltType)): ndarray(rank, eltType) {
    return a;
}

operator +(a: ndarray(?rank,?eltType),b: ndarray(rank,eltType)): ndarray(rank,eltType) {
    const dom = a.domain;

    const ref aData = a.data;
    const ref bData = b.data;
    var c: ndarray(rank,eltType) = new ndarray(dom,eltType);
    ref cData = c.data;
    // @assertOnGpu
    forall i in dom.every() do
        cData[i] = aData[i] + bData[i];
    return c;
}

operator *(a: ndarray(?rank,?eltType),b: ndarray(rank,eltType)): ndarray(rank,eltType) {
    const dom = a.domain;
    var c: ndarray(rank,eltType) = new ndarray(a.domain,eltType);
    ref cData = c.data;
    const ref aData = a.data;
    const ref bData = b.data;
    // @assertOnGpu
    forall i in dom.every() do
        cData[i] = aData[i] * bData[i];
    return c;
}

operator -(a: ndarray(?rank, ?eltType)): ndarray(rank, eltType) {
    const dom = a.domain;
    var negged = new ndarray(dom, eltType);
    ref negData = negged.data;
    const ref data = a.data;
    forall i in dom.every() {
        negData[i] = data[i];
    }

    return negged;
}

operator -(a: ndarray(?rank,?eltType),b: ndarray(rank,eltType)): ndarray(rank,eltType) {
    const dom = a.domain;
    var c: ndarray(rank,eltType) = new ndarray(a.domain,eltType);
    ref cData = c.data;
    const ref aData = a.data;
    const ref bData = b.data;
    // @assertOnGpu
    forall i in dom.every() do
        cData[i] = aData[i] - bData[i];
    return c;
}

operator /(a: ndarray(?rank,?eltType),b: ndarray(rank,eltType)): ndarray(rank,eltType) {
    const dom = a.domain;
    var c: ndarray(rank,eltType) = new ndarray(a.domain,eltType);
    ref cData = c.data;
    const ref aData = a.data;
    const ref bData = b.data;
    // @assertOnGpu
    forall i in dom.every() do
        cData[i] = aData[i] / bData[i];
    return c;
}

inline proc type ndarray.valueLike(a: ndarray(?rank,?eltType),value: eltType): ndarray(rank,eltType) do
    return new ndarray(eltType=eltType,dom=a.domain,fill=value);

// a * c
inline proc type ndarray.scalarMapOp(param op: string, a: ndarray(?rank,?eltType),c: eltType): ndarray(rank,eltType) {
    const dom = a.domain;
    var u: ndarray(rank,eltType) = new ndarray(a.domain,eltType);
    ref uData = u.data;
    const ref aData = a.data;
    forall i in dom.every() {
        select op {
            when "+" do 
                uData[i] = aData[i] + c;
            when "-" do 
                uData[i] = aData[i] - c;
            when "*" do 
                uData[i] = aData[i] * c;
            when "/" do 
                uData[i] = aData[i] / c;
            otherwise do 
                compilerError("Unknown operator ", op);
        }
    }
    return u;
}

// c * a
inline proc type ndarray.scalarMapOp(param op: string, c: ?eltType, a: ndarray(?rank,eltType)): ndarray(rank,eltType) {
    const dom = a.domain;
    var u: ndarray(rank,eltType) = new ndarray(a.domain,eltType);
    ref uData = u.data;
    const ref aData = a.data;
    forall i in dom.every() {
        select op {
            when "+" do 
                uData[i] = c + aData[i];
            when "-" do 
                uData[i] = c - aData[i];
            when "*" do 
                uData[i] = c * aData[i];
            when "/" do 
                uData[i] = c / aData[i];
            otherwise do 
                compilerError("Unknown operator ", op);
        }
    }
    return u;
}

// Left A with right C
inline proc type ndarray.scalarMapOp(param op: string, a: ndarray(?rank,?eltType),c: ?scalarType): ndarray(rank,eltType)
        where isNumericType(scalarType) && scalarType != eltType do
    return ndarray.scalarMapOp(op,a,c : eltType);

// Left C with right A
inline proc type ndarray.scalarMapOp(param op: string, c: ?scalarType, a: ndarray(?rank,?eltType)): ndarray(rank,eltType)
        where isNumericType(scalarType) && scalarType != eltType do
    return ndarray.scalarMapOp(op,c : eltType,a);


operator +(a: ndarray(?rank,?eltType),c: ?scalarType): ndarray(rank,eltType)
        where isNumericType(scalarType) do
    return ndarray.scalarMapOp("+",a,c);

operator +(c: ?scalarType,a: ndarray(?rank,?eltType)): ndarray(rank,eltType)
        where isNumericType(scalarType) do
    return ndarray.scalarMapOp("+",c,a);

operator -(a: ndarray(?rank,?eltType),c: ?scalarType): ndarray(rank,eltType)
        where isNumericType(scalarType) do
    return ndarray.scalarMapOp("-",a,c);

operator -(c: ?scalarType,a: ndarray(?rank,?eltType)): ndarray(rank,eltType)
        where isNumericType(scalarType) do
    return ndarray.scalarMapOp("-",c,a);

operator *(a: ndarray(?rank,?eltType),c: ?scalarType): ndarray(rank,eltType)
        where isNumericType(scalarType) do
    return ndarray.scalarMapOp("*",a,c);

operator *(c: ?scalarType,a: ndarray(?rank,?eltType)): ndarray(rank,eltType)
        where isNumericType(scalarType) do
    return ndarray.scalarMapOp("*",c,a);

operator /(a: ndarray(?rank,?eltType),c: ?scalarType): ndarray(rank,eltType)
        where isNumericType(scalarType) do
    return ndarray.scalarMapOp("/",a,c);

operator /(c: ?scalarType,a: ndarray(?rank,?eltType)): ndarray(rank,eltType)
        where isNumericType(scalarType) do
    return ndarray.scalarMapOp("/",c,a);


// operator +(a: remote(ndarray(?rank,?eltType)),b: remote(ndarray(rank,eltType))): remote(ndarray(rank,eltType)) {
//     const device = a.device;
//     var c = new remote(ndarray(rank,eltType),device);
//     on device {
//         ref A = a.localAccess();
//         ref B = b.localAccess();
//         ref C = c.localAccess();
//         C = a.localAccess() + b.localAccess();
//     }
//     return c;
// }

proc type ndarray.convolve(features: ndarray(3,?eltType),kernel: ndarray(4,eltType), stride: int) do
    return ndarray.convolve(features,kernel,stride,padding = (0,0));

proc type ndarray.convolve(features: ndarray(3,?eltType),kernel: ndarray(4,eltType), bias: ndarray(1,eltType), stride: int, padding: 2*int): ndarray(3,eltType) {
    const (channels,inHeight,inWidth) = features.shape;
    const (filters,channels_,kernelHeight,kernelWidth) = kernel.shape;
    const (filters_,) = bias.shape;
    const (paddingHeight,paddingWidth) = padding;
    if channels != channels_ then halt("Channels must match. ", features.shape , " ", kernel.shape);

    const outHeight: int = ((inHeight - kernelHeight) / stride) + 1 + (paddingHeight * 2);
    const outWidth: int = ((inWidth - kernelWidth) / stride) + 1 + (paddingWidth * 2);
    const outShape = (filters,outHeight,outWidth);
    const outDom = util.domainFromShape((...outShape));

    const ref fet = features.data;
    const ref ker = kernel.data;

    var outFeatures = new ndarray(outDom,eltType);
    ref dat = outFeatures.data;

    inline proc fastKernel(param kernelSize: int) {
        forall (f,h_,w_) in outDom.every() {
            const hi = h_ * stride;
            const wi = w_ * stride;
            var sum: eltType = 0;
            for c in 0..<channels {
                for param kh in 0..<kernelSize {
                    for param kw in 0..<kernelSize {
                        sum += fet[c,hi + kh, wi + kw] * ker[f,c,kh,kw];
                    }
                }
            }
            dat[f,h_ + paddingHeight,w_ + paddingWidth] = sum;
        }
    }

    inline proc slowKernel() {
        const kernelChanD = util.domainFromShape(channels,kernelHeight,kernelWidth);
        forall (f,h_,w_) in outDom.every() {
            const hi = h_ * stride;
            const wi = w_ * stride;
            var sum: eltType = 0;
            if util.targetGpu() then
                for c in 0..<channels do
                    for kh in 0..<kernelHeight do
                        for kw in 0..<kernelWidth do
                            sum += fet[c,hi + kh, wi + kw] * ker[f,c,kh,kw];
            else
                for (c,kh,kw) in kernelChanD do
                    sum += fet[c,hi + kh, wi + kw] * ker[f,c,kh,kw];
            dat[f,h_ + paddingHeight,w_ + paddingWidth] = sum;
        }
    }

    select (kernelHeight,kernelWidth) {
        when (3,3) do 
            fastKernel(3);
        when (5,5) do
            fastKernel(5);
        when (7,7) do
            fastKernel(7);
        when (9,9) do
            fastKernel(9);
        when (11,11) do
            fastKernel(11);
        otherwise do
            slowKernel();
    }

    return outFeatures;
}


proc type ndarray.convolve(features: ndarray(3,?eltType),kernel: ndarray(4,eltType), stride: int, padding: int): ndarray(3,eltType) {
    const (channels, inHeight, inWidth) = features.shape;
    const (filters, channels_, kernelHeight, kernelWidth) = kernel.shape;
    if channels != channels_ then halt("Channels must match.");

    // Calculate the dimensions of the output feature map
    const outHeight: int = ((inHeight + 2 * padding - kernelHeight) / stride) + 1;
    const outWidth: int = ((inWidth + 2 * padding - kernelWidth) / stride) + 1;
    const outShape = (filters, outHeight, outWidth);
    const outDom = util.domainFromShape((...outShape));
    var outFeatures = new ndarray(outDom, eltType);

    // Create a padded feature map
    const paddedHeight = inHeight + 2 * padding;
    const paddedWidth = inWidth + 2 * padding;
    const paddedDom = util.domainFromShape(channels, paddedHeight, paddedWidth);
    var paddedFeatures = new ndarray(paddedDom, eltType);
    ref paddedData = paddedFeatures.data;
    ref fet = features.data;

    // Initialize the padded feature map with zeros
    paddedData = 0;

    // Copy the original feature map into the center of the padded feature map
    forall (c, h, w) in features.domain {
        paddedData[c, h + padding, w + padding] = fet[c, h, w];
    }

    ref dat = outFeatures.data;
    ref ker = kernel.data;

    const kernelChanD = util.domainFromShape(channels, kernelHeight, kernelWidth);

    // Perform the convolution on the padded feature map
    forall (f, h_, w_) in outDom.every() {
        const hi: int = h_ * stride;
        const wi: int = w_ * stride;
        var sum: eltType = 0;
        for j in 0..<kernelChanD.size {
            const (c, kh, kw) = kernelChanD.indexAt(j);
            sum += paddedData[c, hi + kh, wi + kw] * ker[f, c, kh, kw];
        }
        dat[f, h_, w_] = sum;
    }

    return outFeatures;
}

proc type ndarray.convolve(features: ndarray(3,?eltType),kernel: ndarray(4,eltType), bias: ndarray(1,eltType), stride: int): ndarray(3,eltType) {
    const (channels,inHeight,inWidth) = features.shape;
    const (filters,channels_,kernelHeight,kernelWidth) = kernel.shape;
    const (filters_,) = bias.shape;
    if channels != channels_ then halt("Channels must match. ", features.shape , " ", kernel.shape);
    if filters != filters_ then halt("Bias and filters must match.");

    const outHeight: int = ((inHeight - kernelHeight) / stride) + 1;
    const outWidth: int = ((inWidth - kernelWidth) / stride) + 1;
    const outShape = (filters,outHeight,outWidth);
    const outDom = util.domainFromShape((...outShape));

    const ref fet = features.data;
    const ref ker = kernel.data;
    const ref bis = bias.data;

    var outFeatures = new ndarray(outDom,eltType);
    ref dat = outFeatures.data;

    inline proc fastKernel(param kernelSize: int) {
        forall (f,h_,w_) in outDom.every() {
            const hi = h_ * stride;
            const wi = w_ * stride;
            var sum: eltType = 0;
            for c in 0..<channels {
                for param kh in 0..<kernelSize {
                    for param kw in 0..<kernelSize {
                        sum += fet[c,hi + kh, wi + kw] * ker[f,c,kh,kw];
                    }
                }
            }
            dat[f,h_,w_] = sum + bis[f];
        }
    }

    inline proc slowKernel() {
        const kernelChanD = util.domainFromShape(channels,kernelHeight,kernelWidth);
        forall (f,h_,w_) in outDom.every() {
            const hi = h_ * stride;
            const wi = w_ * stride;
            var sum: eltType = 0;
            if util.targetGpu() then
                for c in 0..<channels do
                    for kh in 0..<kernelHeight do
                        for kw in 0..<kernelWidth do
                            sum += fet[c,hi + kh, wi + kw] * ker[f,c,kh,kw];
            else
                for (c,kh,kw) in kernelChanD do
                    sum += fet[c,hi + kh, wi + kw] * ker[f,c,kh,kw];
            dat[f,h_,w_] = sum + bis[f];
        }
    }

    select (kernelHeight,kernelWidth) {
        when (3,3) do
            fastKernel(3);
        when (5,5) do
            fastKernel(5);
        when (7,7) do
            fastKernel(7);
        when (9,9) do
            fastKernel(9);
        when (11,11) do
            fastKernel(11);
        otherwise do
            slowKernel();
    }
    return outFeatures;
}

proc type ndarray.convolve(features: ndarray(3,?eltType), kernel: ndarray(4,eltType), bias: ndarray(1,eltType), stride: int, padding: int): ndarray(3,eltType) {
    const (channels, inHeight, inWidth) = features.shape;
    const (filters, channels_, kernelHeight, kernelWidth) = kernel.shape;
    const (filters_,) = bias.shape;
    if channels != channels_ then halt("Channels must match. ", features.shape , " ", kernel.shape);
    if filters != filters_ then halt("Bias and filters must match.");

    // Calculate the dimensions of the output feature map
    const outHeight: int = ((inHeight + 2 * padding - kernelHeight) / stride) + 1;
    const outWidth: int = ((inWidth + 2 * padding - kernelWidth) / stride) + 1;
    const outShape = (filters, outHeight, outWidth);
    const outDom = util.domainFromShape((...outShape));

    // Create a padded feature map
    const paddedHeight = inHeight + 2 * padding;
    const paddedWidth = inWidth + 2 * padding;
    const paddedDom = util.domainFromShape(channels, paddedHeight, paddedWidth);
    var paddedFeatures = new ndarray(paddedDom, eltType);
    ref paddedData = paddedFeatures.data;
    ref fet = features.data;

    // Initialize the padded feature map with zeros
    paddedData = 0;

    // Copy the original feature map into the center of the padded feature map
    forall (c, h, w) in features.domain {
        paddedData[c, h + padding, w + padding] = fet[c, h, w];
    }

    const ref ker = kernel.data;
    const ref bis = bias.data;

    var outFeatures = new ndarray(outDom, eltType);
    ref dat = outFeatures.data;

    inline proc fastKernel(param kernelSize: int) {
        forall (f, h_, w_) in outDom.every() {
            const hi = h_ * stride;
            const wi = w_ * stride;
            var sum: eltType = 0;
            for c in 0..<channels {
                for param kh in 0..<kernelSize {
                    for param kw in 0..<kernelSize {
                        sum += paddedData[c, hi + kh, wi + kw] * ker[f, c, kh, kw];
                    }
                }
            }
            dat[f, h_, w_] = sum + bis[f];
        }
    }

    inline proc slowKernel() {
        const kernelChanD = util.domainFromShape(channels, kernelHeight, kernelWidth);
        forall (f, h_, w_) in outDom.every() {
            const hi = h_ * stride;
            const wi = w_ * stride;
            var sum: eltType = 0;
            if util.targetGpu() then
                for c in 0..<channels do
                    for kh in 0..<kernelHeight do
                        for kw in 0..<kernelWidth do
                            sum += paddedData[c, hi + kh, wi + kw] * ker[f, c, kh, kw];
            else
                for (c, kh, kw) in kernelChanD do
                    sum += paddedData[c, hi + kh, wi + kw] * ker[f, c, kh, kw];
            dat[f, h_, w_] = sum + bis[f];
        }
    }

    select (kernelHeight, kernelWidth) {
        when (3, 3) do
            fastKernel(3);
        when (5, 5) do
            fastKernel(5);
        when (7, 7) do
            fastKernel(7);
        when (9, 9) do
            fastKernel(9);
        when (11, 11) do
            fastKernel(11);
        otherwise do
            slowKernel();
    }
    return outFeatures;
}


proc type ndarray.maxPool(features: ndarray(3, ?eltType), poolSize: int) do return this.maxPool(features,poolSize,poolSize);
proc type ndarray.maxPool(features: ndarray(3,?eltType),poolSize: int, stride: int, padding: int = 0, dilation: int = 1): ndarray(3,eltType) {
    const (channels, height, width) = features.shape;

    // Calculate the effective pool size considering dilation
    // effectivePoolSize = poolSize + (poolSize - 1) * (dilation - 1)
    // Ex: poolSize = 2, dilation = 1 -> effectivePoolSize = 2
    // Ex: poolSize = 3, dilation = 2 -> effectivePoolSize = 5
    // The dilation controls the stride within the pooling window
    const effectivePoolSize = poolSize + (poolSize - 1) * (dilation - 1);

    // Calculate the new height and width after padding
    // Add zeroes to the height and width only, not channels
    const paddedHeight = height + 2 * padding;
    const paddedWidth = width + 2 * padding;

    // Calculate the new height and width after pooling
    // hOut = floor((hIn + 2 * padding - dilation *(poolSize -1 ) - 1) / stride) + 1
    // wOut = floor((wIn + 2 * padding - dilation *(poolSize -1 ) - 1) / stride) + 1
    const newHeight: int = Math.floor((paddedHeight - dilation * (poolSize - 1) - 1) / stride):int + 1;
    const newWidth: int = Math.floor((paddedWidth - dilation * (poolSize - 1) - 1) / stride):int + 1;

    // Create a new domain and ndarray for the result
    const dom = util.domainFromShape(channels, newHeight, newWidth);
    var pool = new ndarray(dom, eltType);
    ref dat = pool.data;
    ref fet = features.data;

    // Create a domain for the pooling window
    const poolDom = util.domainFromShape(poolSize, poolSize);

    // Perform max pooling with padding and dilation
    forall (c, h, w) in dom.every() {
        const hs = h * stride - padding;
        const ws = w * stride - padding;
        var mx: eltType = -Math.inf: eltType; // Initialize to negative infinity for max pooling
        for (ph, pw) in poolDom {
            const hIndex = hs + ph * dilation;
            const wIndex = ws + pw * dilation;
            if hIndex >= 0 && hIndex < height && wIndex >= 0 && wIndex < width {
                const x: eltType = fet[c, hIndex, wIndex];
                mx = Math.max(x, mx);
            }
        }
        dat[c, h, w] = mx;
    }
    return pool;
}

// adaptiveAvgPool2d
proc type ndarray.adaptiveAvgPool2d(features: ndarray(3, ?eltType), outputSize: int): ndarray(3, eltType) {
    const (channels, height, width) = features.shape;
    const newHeight = outputSize;
    const newWidth = outputSize;
    const dom = util.domainFromShape(channels, newHeight, newWidth);
    var pool = new ndarray(dom, eltType);
    ref dat = pool.data;
    ref fet = features.data;

    // Calculate the size of each pooling region
    const poolHeight = (height + newHeight - 1) / newHeight;
    const poolWidth = (width + newWidth - 1) / newWidth;

    // Perform adaptive average pooling
    forall (c, h, w) in dom.every() {
        const hs = (h * height) / newHeight;
        const ws = (w * width) / newWidth;
        const he = ((h + 1) * height + newHeight - 1) / newHeight;
        const we = ((w + 1) * width + newWidth - 1) / newWidth;

        var sum: eltType = 0;
        var count: int = 0;
        for ph in hs..<he {
            for pw in ws..<we {
                if ph < height && pw < width {
                    sum += fet[c, ph, pw];
                    count += 1;
                }
            }
        }
        dat[c, h, w] = sum / count;
    }
    return pool;
}


proc type ndarray.sqrt(array: ndarray(?rank,?eltType)): ndarray(rank,eltType) {
    const dom = array.domain;
    var sqrtArr = new ndarray(dom,eltType);
    ref sqrtData = sqrtArr.data;
    const ref thisData = array.data;
    // @assertOnGpu
    forall i in dom.every() {
        sqrtData[i] = Math.sqrt(thisData[i]);
    }
    return sqrtArr;
}

proc type ndarray.matvecmul(mat: ndarray(2,?eltType),vec: ndarray(1,eltType)): ndarray(1,eltType) {
    const (m,n) = mat.shape;
    const (n_,) = vec.shape;
    assert(n == n_, "Vector and matrix must be the same shape.");
    const dom = util.domainFromShape(m);
    var u = new ndarray(dom,eltType);
    ref matD = mat.data;
    ref vecD = vec.data;
    ref uD = u.data;
    // @assertOnGpu
    forall i in 0..<m {
        var sum: eltType;
        for j in 0..<n {
            sum += matD[i,j] * vecD[j];
        }
        uD[i] = sum;
    }
    return u;
}

proc type ndarray.batchNorm(
    features: ndarray(?rank,?eltType),
    weight: ndarray(1,eltType),
    bias: ndarray(1, eltType),
    movingAvg: ndarray(1, eltType),
    movingVar: ndarray(1, eltType),
    n: int // num_features
): ndarray(rank,eltType) {
    // writeln("IN ndarray.batchNorm");
    if rank < 2 then halt("Rank must be greater than 2");
    if rank > 4 then halt("Rank must be less than 4");
    const fshape = features.shape;

    ref f = features.data;
    ref w = weight.data;
    ref b = bias.data;
    ref a = movingAvg.data;
    ref v = ndarray.sqrt(movingVar).data;

    var outDom = util.domainFromShape((...fshape));
    var outFeatures = new ndarray(outDom,eltType);
    ref dat = outFeatures.data;

    forall idx in outDom.every() {
        var c = idx[1];
        dat[idx] = w[c]*((f[idx]-a[c])/v[c])+b[c];
    }

    return outFeatures;

}


inline proc type ndarray.fromRanges(type eltType = real, rngs: range...?rank): ndarray(rank,eltType) {
    const dom_ = {(...rngs)};
    const dom = util.domainFromShape((...dom_.shape));
    var a = new ndarray(dom,eltType);
    ref aData = a.data;
    // @assertOnGpu
    forall i in 0..<dom.size with (ref a) {
        const idx = dom.indexAt(i);
        aData[idx] = i : eltType;
    }
    return a;
}

proc type ndarray.nllLoss(
    input: ndarray(2,?eltType), 
    target: ndarray(1,eltType), 
    weight: ndarray(1, eltType),
    ignoreIndex: int = -1,
    red: bool = true,
    reduction: string = "mean"
): ndarray(1,eltType) {
    const (N,C) = input.shape;
    assert(target.shape[0] == N, "Target shape must match batch size.");
    assert(weight.shape[0] == C, "Weights shape must match number of classes.");
    
    const dom = util.domainFromShape(N);
    var loss = new ndarray(dom, eltType);
    ref x = input.data;
    ref y = target.data;
    ref w = weight.data;
    ref lossD = loss.data;
    var wynSum: real = 0.0;

    forall n in 0..<N with (+ reduce wynSum) {
        const yn: int = y[n]:int;
        if yn == ignoreIndex {
            lossD[n] = 0.0;
        }
        else {
            lossD[n] = -w[yn]*x[n,yn];
            wynSum += w[yn];
        }
    }

    if !red then return loss;
    if reduction == "mean" then return loss.sum(0) / wynSum;
    if reduction == "sum" then return loss.sum(0);
    halt("Invalid reduction mode: " + reduction);
}

module ndarrayRandom {
    private import Random;

    var globalSeedSetFlag: bool = false;
    var globalSeed: int = -1;

    proc seed: int {
        if globalSeedSetFlag {
            var rs = new Random.randomStream(int,globalSeed);
            globalSeed = rs.next();
            return globalSeed;
        } else {
            var rs = new Random.randomStream(int);
            return rs.next();
        }
    }

    proc setGlobalSeed(newSeed: int) {
        globalSeedSetFlag = true;
        var rs = new Random.randomStream(int,newSeed);
        globalSeed = rs.next();
    }

    proc getRandomStream(type eltType): Random.randomStream(eltType) {
        if globalSeedSetFlag {
            return new Random.randomStream(eltType,seed);
        } else {
            return new Random.randomStream(eltType);
        }
    }
}

proc type ndarray.setGlobalRandomSeed(seed: int) do
    ndarrayRandom.setGlobalSeed(seed);

proc type ndarray.getNextSeed(): int do
    return ndarrayRandom.seed;

proc type ndarray.getRandomStream(type eltType): Random.randomStream(eltType) do
    return ndarrayRandom.getRandomStream(eltType);

proc type ndarray.randomArray(
    shape: int...?rank,
    type eltType = defaultEltType,
    in rs: Random.randomStream(eltType) = ndarray.getRandomStream(eltType)): ndarray(rank,eltType) {
    const dom = util.domainFromShape((...shape));
    return new ndarray(eltType,rs,dom);
}

proc type ndarray.random(shape: int...?rank,type eltType = defaultEltType): ndarray(rank,eltType) do
    return ndarray.randomArray((...shape),eltType,ndarray.getRandomStream(eltType));

proc type ndarray.random(shape: int...?rank): ndarray(rank,defaultEltType) do
    return ndarray.random((...shape),eltType = defaultEltType);

proc type ndarray.random(shape: ?rank*int,type eltType = defaultEltType,seed: int = ndarray.getNextSeed()): ndarray(rank,eltType) do
    return ndarray.randomArray((...shape),eltType,new Random.randomStream(eltType,seed));


proc type ndarray.loadImage(imagePath: string, type eltType = defaultEltType): ndarray(3,eltType) {
    import Image;

    param chanBits = Image.bitsPerColor; 
    param chanGran = 2 ** chanBits; // Channel granularity for each channel
    const cgInEltType: eltType = chanGran : eltType;

    inline proc getColorFromPixel(pixel: Image.pixelType, param offset: int) {
        return (pixel >> Image.colorOffset(offset)) & Image.colorMask;
    }

    inline proc getChannelValue(pixel: Image.pixelType, param offset: int): eltType {
        if isRealType(eltType) {
            const color: eltType = getColorFromPixel(pixel,offset);
            return color / cgInEltType;
        } else {
            compilerError("Only real types are supported for now.");
        }
    }

    const pixelFormat = (Image.rgbColor.red,Image.rgbColor.green,Image.rgbColor.blue);

    const imgType = util.getImageType(imagePath);
    const pixelData = Image.readImage(imagePath,format=imgType);
    const (height,width) = pixelData.shape;

    const imgDom = util.domainFromShape(3,height,width);
    var img = new ndarray(imgDom,eltType);
    ref imgData = img.data;

    forall (pixel,(i,j)) in zip(pixelData,pixelData.domain) do
        for param c in 0..<pixelFormat.size do
            imgData[c,i,j] = getChannelValue(pixel,c); // getColorFromPixel(pixel,pixelFormat[c]);

    return img;
}

proc ref ndarray.saveImage(imagePath: string) where rank == 3 {

    // compilerWarning("I have not implemented ndarray.saveImage");
    import Image;

    param chanBits = Image.bitsPerColor; 
    param chanGran = 2 ** chanBits; // Channel granularity for each channel
    const cgInEltType: eltType = chanGran : eltType;

    const imgType = util.getImageType(imagePath);

    inline proc getColorAsPixel(color: Image.pixelType, param offset: int) {
        return (color & Image.colorMask) << Image.colorOffset(offset);
    }

    inline proc getChannelValue(channel: eltType, param offset: int): Image.pixelType {
        if isRealType(eltType) {
            const pixel = (channel * cgInEltType) : Image.pixelType;
            return getColorAsPixel(pixel,offset);
        } else {
            compilerError("Only real types are supported for now.");
        }
    }

    const pixelFormat = (Image.rgbColor.red,Image.rgbColor.green,Image.rgbColor.blue);

    const (_,height,width) = this.shape;
    const pixelDom = util.domainFromShape(height,width);
    var pixelData: [pixelDom] Image.pixelType;
    ref imgData = this.data;

    forall (i,j) in pixelDom {
        var pixel: Image.pixelType;
        for param c in 0..<pixelFormat.size do
            pixel |= getChannelValue(imgData[c,i,j],c);
        pixelData[i,j] = pixel;
    }

    Image.writeImage(imagePath,format=imgType,pixels=pixelData);
}

// For printing. 
proc ndarray.serialize(writer: IO.fileWriter(locking=false, IO.defaultSerializer),ref serializer: IO.defaultSerializer) throws {
    
    const format = util.roundingFormat(this.data);
    const name = "ndarray";
    const header = name + "(";
    const indent = (" " * name.size) + (" " * this.rank);
    const dataStr = util.prettyPrintArray(indent,format,this.flatten().data,this.domain.shape);
    writer.write(header);
    writer.write(dataStr);
    writer.write(",\n       shape = ",this.domain.shape);
    writer.write(",\n       rank = ",this.rank);
    writer.writeln(")");
}

proc ref ndarray.read(fr: IO.fileReader(?)) throws {
    var r = fr.read(int);
    if r != rank then
        err("Error reading tensor: rank mismatch.", r , " != this." , rank);
    var s = this.shape;
    for i in 0..#rank do
        s[i] = fr.read(int);
    var d = util.domainFromShape((...s));
    this._domain = d;
    for i in d do
        this.data[i] = fr.read(eltType);
}


proc ndarray.write(fw: IO.fileWriter(?)) throws {
    fw.write(rank);
    for s in data.domain.shape do
        fw.write(s:int);
    for i in data.domain do
        fw.write(data[i]);
}

class _tensor_resource {
    param rank: int;
    type eltType = real(64);
    var data: remote(ndarray(rank,eltType));


}

// Some examples. 
// const dom = util.emptyDomain(2);
// writeln(dom);

// printType(ndarray(1));
// var A: ndarray(1);
// A._domain = {0..#3};
// // A.data = [1.0,2.0,3.0];
// A = [1.0,2.0,3.0];
// writeln(A);

// use GpuDiagnostics;

// startGpuDiagnostics();
// startVerboseGpu();

// var device = here.gpus[0];

// on device {
//     var arr = [1.0,2.0,3.0];

//     var A: ndarray(1) = arr;
//     writeln("A -> ",A);

//     var B: ndarray(1) = new ndarray(arr);
//     // B = [1.0,2.0,3.0];

//     writeln("B -> ",B.domain);

//     var dom = {0..#3,0..#5};

//     var C: ndarray(2) = new ndarray(dom);
//     writeln("C -> ",C);
// }

// var A: ndarray(1) = [1.0,2.0,3.0];
// writeln(A);

// var B: ndarray(1) = new ndarray([1.0,2.0,3.0]);
// // B = [1.0,2.0,3.0];

// writeln(B.domain);

// var C: ndarray(2) = new ndarray({0..#3,0..#5});
// var c = 0;
// for i in C.domain {
//     C.data[i] = c;
//     c += 1;
// }
// writeln(C);

// C = C.reshape({0..#5,0..#3});
// writeln(C);

// writeln(C[0,0]);

// C[0,0] = 70.0;

// var D = C[{0..1,0..1}];

// C[{0..1,0..1}]=2.0;


// writeln(D,D.type:string);
// writeln(C,C.type:string);

// var E = C.slice(1,..);
// writeln(E);


// type T = (int,real);
// writeln(T:string,isTupleType(T));

// end examples. 



proc type ndarray.fullOuter(a: ndarray(?rankA,?eltType), b: ndarray(?rankB, eltType)): ndarray(rankA + rankB, eltType) {
    param rankC = rankA + rankB;
    var newShape: rankC * int = ((...a.shape), (...b.shape));
    const domC = util.domainFromShape((...newShape));
    var c: ndarray(rankC,eltType) = new ndarray(domC,eltType);
    ref cData = c.data;
    foreach i in domC.each {
        const aIdx = i.slice(0,rankA); // i(0..<rankA);
        const bIdx = i.slice(rankA+1,rankB); // i(rankA..<rankB);
        cData[i] = a.data[aIdx] * b.data[bIdx];
    }
    return c;
}

proc type ndarray.contract(a: ndarray(?rankA,?eltType), b: ndarray(?rankB, eltType), param axisA: int, param axisB: int) {
    const axA = a.domain.dim(axisA);
    const axB = b.domain.dim(axisB);
    if axA != axB then halt("Not same axes!");

    param newRank = a.rank + b.rank - 2;
    const contractedShapeA = a.shape.removeIdx(axisA);
    const contractedShapeB = b.shape.removeIdx(axisB);


    const newShape = ((...contractedShapeA),(...contractedShapeB));
    const dom = util.domainFromShape((...newShape));
    var c: ndarray(newRank, eltType) = new ndarray(dom,eltType);
    foreach i in c.domain.each with (ref c) {
        var idxA: a.rank * int = i.slice(0,contractedShapeA.size).insertIdx(axisA,0);
        var idxB: b.rank * int = i.slice(contractedShapeA.size,newRank).insertIdx(axisB,0);
        var sum: eltType = 0;
        for (ai,bi) in {a.domain.dim(axisA),b.domain.dim(axisB)} {
            idxA(axisA) = ai;
            idxB(axisB) = bi;
            sum += a.data[idxA] * b.data[idxB];
        }
        c.data[i] = sum;
    }
    return c;
}

proc splitAt(param s: string, param del: string, param idx: int = 0) param {
    if s[idx] == del {
        return "";
    } else {
        return s[idx] + splitAt(s,del,idx + 1);
    }
}

proc getFirstIdx(param s: string, param del: string, param idx: int = 0) param {
    if s[idx] == del {
        return idx;
    } else {
        return getFirstIdx(s,del,idx + 1);
    }
}

proc sliceGeneric(type t, param start: int, param stop: int, param s: t, param idx: int = start) param {
    compilerAssert(start <= stop);
    compilerAssert(stop <= s.size);
    if start <= idx && idx < stop {
        return s[idx] + slice(start,stop,s,idx + 1);
    } else {
        param zero: t;
        return zero;
    }
}

proc slice(param start: int, param stop: int, param s: string, param idx: int = start) param {
    compilerAssert(start <= stop);
    compilerAssert(stop <= s.size);
    if start <= idx && idx < stop {
        return s[idx] + slice(start,stop,s,idx + 1);
    } else {
        param zero: string;
        return zero;
    }
}

proc take(param count: int, param s: string) param do
    return slice(0,count,s);

proc drop(param count: int, param s: string) param do
    return slice(count,s.size,s);



proc type ndarray.einsum(param subscripts: string,a: ndarray(?rankA,?eltType), b: ndarray(?rankB, eltType)) {

    for param i in 0..<subscripts.size {
        param c = subscripts[i];
        // writeln(c);
    }
    param fst = subscripts.takeUntil(",");
    param subscripts_1 = subscripts.drop(fst.size + 1);
    param snd = splitAt(subscripts_1,"-");
    param subscripts_2 = subscripts_1.drop(snd.size + 2);
    // param thd 
    // param fth
    // param vth
    // Cool three letter names for 3,4,5

    for param i in 0..<fst.size {
        param ci = fst[i];
        if fst.countOccurrences(ci) != subscripts_2.countOccurrences(ci) {
            for param j in 0..<snd.size {
                param cj = snd[j];
                if snd.countOccurrences(cj) != subscripts_2.countOccurrences(cj) {
                    return ndarray.contract(a,b,i,j);
                }
            }
        }
    }
    // compilerError("Must sum across one axis!");
    return a;
}


proc ndarray.softmax(): ndarray(this.rank, this.eltType)
    where isSubtype(this.eltType, real)
{
    const dom = this.domain;
    const ref thisData = this.data;

    var denom: this.eltType = 0.0;
    forall i in dom.every() with (+ reduce denom) {
        denom += Math.exp(thisData[i]);
    }

    var softmaxxed = new ndarray(this.eltType, dom);
    ref softmaxData = softmaxxed.data;
    forall i in dom.every() {
        softmaxData[i] = Math.exp(thisData[i]) / denom;
    }

    return softmaxxed;
}


proc ndarray.softmin(): ndarray(this.rank, this.eltType)
    where isSubtype(this.eltType, real)
{
    return (-this).softmax();
}


proc ndarray.dropout(): ndarray(this.rank, this.eltType) {
    const randomData: int[this.domain];
    Random.fillRandom(randomData, 0, 1);

    ref thisData = this.data;
    forall i in this.domain.every() {
        thisData[i] *= randomData[i];
    }

    return this;
}


proc main() {
    // More examples. 
    writeln("Hello!");
    // {
    //     var A = ndarray.fromRanges(real, 0..<5);
    //     var B = ndarray.fromRanges(real, 0..<3);

    //     A.data += 1;
    //     B.data += 1;

    //     writeln(A);
    //     writeln(B);
        
    //     var C = ndarray.fullOuter(A,B);
    //     writeln(C);

    //     var t = (1,2,3,4,5,6);
    //     writeln(t);
    //     writeln(t.removeIdx(3));
    // }

    // var A = ndarray.fromRanges(real, 0..<5,0..<3);
    // var B = ndarray.fromRanges(real, 0..<3,0..<7);

    // A.data += 1;
    // B.data += 1;

    // var D = ndarray.contract(A,B,1,0);
    // writeln(D);

    // var E = ndarray.einsum("ij,kh->ih",A,B);
    // writeln(E);
    // E[1,1] = 2;


    var A = ndarray.fromRanges(real, 0..<5,0..<3);
    var B = ndarray.fromRanges(real, 0..<5,0..<3);

    writeln(A + B);


    // param r = 0..<3;
    // writeln(r);
}

}
