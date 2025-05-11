use Tensor;
use CTypes;

proc main(args: [] string) {
    writeln("Hello, world!");

    // cobegin {
    //     for i in 0..<100 {
    //         begin Bridge.splitLoop(i,100);
    //     }
    // }
    var n: int(64) = 0;
    var nr = c_ptrTo(n);
    cobegin {
        begin Bridge.splitLoopFiller(1000000,nr);

        for i in 0..<10 {
            writeln("Hello from ", nr.deref());
        }

    }


    Bridge.showWebcam();

    writeln("Done!");
}