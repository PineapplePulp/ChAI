#include "lib/smol.h"

int main(int argc, char* argv[]) {
    chpl_library_init(argc, argv);

    square(3);

    int64_t array[4] = {1,2,3,4};
    chpl_external_array array_ptr = chpl_make_external_array_ptr(&array,4);
    int64_t sum = sumArray(&array_ptr);
    chpl_free_external_array(array_ptr);
    printf("sum: %d\n", sum);


    int64_t matrix[2][3] = { {1, 4, 2}, {3, 6, 8} };
    chpl_external_array matrix_ptr = chpl_make_external_array_ptr(matrix, 3 * 2);
    printArray(&matrix_ptr);
    chpl_free_external_array(matrix_ptr);


    chpl_library_finalize();

    return 0;
}


