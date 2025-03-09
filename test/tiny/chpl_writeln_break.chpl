import IO;
use IO;
use IO.FormattedIO;

record R : serializable {
    var x: real(32) = -440.37115478515625;
}

// proc R.serialize(writer: IO.fileWriter(locking=false, IO.defaultSerializer),ref serializer: IO.defaultSerializer) {
//     compilerError("R.serialize not implemented");
//     try! throw new Error("Error: x = " + this.x:string);
// }

var A: [0..3] R;
writeln(A);

var B: [0..3] real(32) = A.x;
writeln(B);
// writef("%dr", B);

var C: [0..3] string = [b in B] "%dr".format(b);

writeln(C);

var D: [0..3] string = [b in B] "%1.14dr".format(b);
writeln(D);
