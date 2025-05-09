
use CTypes;

require "mirror.h", "-lmirror";

extern proc run_mirror(): void;


proc main(args: [] string) {
    writeln("Hello, world!");

    run_mirror();
}
