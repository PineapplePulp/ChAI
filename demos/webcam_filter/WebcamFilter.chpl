use Tensor;


proc main(args: [] string) {
    const a = Bridge.captureWebcam(0);

    writeln(a.type:string);
    const b = Bridge.bridgeTensorToArray(3,a);
    writeln(b.shape);
    writeln(b);

}