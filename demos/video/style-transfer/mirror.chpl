
use CTypes;

// require "mirror.h", "-lmirror";
// require "mirror.h libmirror.a";
// require "mirror.h";

// require "mirror.h", "mirror.o";

// g++ -std=c++17 -O2 -fPIC -c mirror.cpp -o mirror.o $(pkg-config --cflags --libs opencv4)
// chpl mirror.h mirror.o mirror.chpl --print-commands --ldflags $(pkg-config --cflags --libs opencv4)

extern proc run_mirror(): int;

extern record cvVideoCapture {}

extern proc get_video_capture(): cvVideoCapture;

proc main(args: [] string) {
    writeln("Hello, world!");

    var x = run_mirror();
    writeln("x: ", x);
    writeln("Done!");
}
