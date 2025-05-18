#include "lib/smol.h"

int main(int argc, char* argv[]) {
    chpl_library_init(argc, argv);

    square(3);

    chpl_library_finalize();

    return 0;
}


