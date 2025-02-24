module Layer {
    private use Tensor;
    private use Network;


    // pragma "unsafe"
    // inline proc myGetField(const ref obj:?t, param name:string) const ref {
    //     use Reflection;
    //     param i = __primitive("field name to num", t, name);
    //     // if i == 0 then
    //     //     compilerError("field ", name, " not found in ", t:string);
    //     return __primitive("field by num", obj, i);
    // }

    inline proc getInstanceMethodPtr(obj: ?t, param fn: string) {
        return __primitive("method call resolves", obj, fn);
    }

    inline proc getInstanceMethodPtr(obj: ?t, param fn: string, args: ?argsType)
            where isTupleType(argsType) {
        return __primitive("method call resolves", obj, fn, (...args));
    }

    proc hasMethod(obj: ?t, param fn: string) param : bool do
        return __primitive("method call resolves",obj,fn);

    proc hasMethod(obj: ?t, param fn: string,) param : bool do
        return __primitive("method call resolves",obj,fn);


    class Activation : Module(?) {
        param activationName: string;
        var activationArgs;

        proc init(type eltType, param activationName: string, const activationArgs: ?argsType)
                where isTupleType(argsType) || isNothingType(argsType) {
            super.init(eltType);
            this.activationName = activationName;
            this.activationArgs = activationArgs;
        }

        proc init(type eltType, param activationName: string) do
            this.init(eltType, activationName, none);
        
        proc init(param activationName: string) do
            this.init(eltType=defaultEltType, activationName, none);


        proc init(param activationName: string, const activationArgs: ?argsType)
                where isTupleType(argsType) do
            this.init(eltType=defaultEltType,activationName,activationArgs);

        override proc forward(input: dynamicTensor(eltType)): dynamicTensor(eltType) {

            writeln("Activation: ",activationName);
            writeln("Activation Args: ",activationArgs);
            if isNothingType(activationArgs.type) then
                writeln("Input method: ",getInstanceMethodPtr(input,activationName));
            else 
                writeln("Input method: ",getInstanceMethodPtr(input,activationName,(...activationArgs)));

            if isNothingType(activationArgs.type) {
                // return __primitive("method call resolves",input,this.activationName);
                writeln(__primitive("method call resolves",input,this.activationName));
            } else {
                compilerError("Wrong branch!");
                // return getInstanceMethodPtr(input,activationName,activationArgs)((...activationArgs));
            }

            // param nf = __primitive("num fields", input);
            // compilerWarning("nf = " + nf:string);
            // writeln("nf = ",nf);
            // for param i in 1..<(nf + 2) {
            //     compilerWarning("Field ", i:string , ": " , __primitive("field num to name", input, i):string);
            //     // writeln("Field ",i,": ",__primitive("field num to name", input, i + 1));
            //     // compilerWarning(i,getFieldName(input.type,i));
            // }




            // use Reflection;
            // param nf = getNumFields(input.type);
            // writeln("Type: ",input.type:string);
            // writeln("Num fields: ", nf);

            // for param i in 0..<nf {
            //     writeln("Field ",i,": ",getFieldName(input.type,i));
            // }
            // writeln(getFieldIndex(input.type,activationName));
            // if isNothingType(activationArgs.type) then
            //     writeln("YEY: ",(activationName,activationArgs)," ",canResolveMethod(input,activationName));
            // else 
            //     writeln("YEY: ",(activationName,activationArgs)," ", canResolveMethod(input,activationName,(...activationArgs)));


            ////// import Reflection;

            // pragma "unsafe"
            // inline proc myGetField(const ref obj:?t, param name:string) const ref {
            //     param i = __primitive("field name to num", t, name);
            //     if i == 0 then
            //         compilerError("field ", name, " not found in ", t:string);
            //     return __primitive("field by num", obj, i);
            // }



            //// param i = __primitive("num fields", input.type);
            //// __primitive("field name to num", t, s);
            //// writeln("Impl fields: ",i);



            // if isNothingType(activationArgs.type) then
            //     if Reflection.canResolveMethod(input,activationName) {
            //         return Reflection.getField(input, activationName)();
            //     }
            //     else {
            //         writeln("Cannot resolve method ",activationName);
            //         return input;
            //     }
            // else 
            //     if Reflection.canResolveMethod(input,activationName,(...activationArgs)) {
            //         return Reflection.getField(input, activationName)((...activationArgs));
            //     }
            //     else {
            //         writeln("Cannot resolve method ",activationName);
            //         return input;
            //     }
            return input;
        }
    }

    // class ResidualBlock : 

}