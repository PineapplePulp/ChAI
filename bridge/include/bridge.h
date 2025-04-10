#ifndef BRIDGE_H
#define BRIDGE_H

#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct tensor_result_t {
    float* data;
    int* sizes;
    int dim;
} tensor_result_t;

int baz(void);

void wrHello(void);

void wrHelloTorch(void);

float sumArray(float* arr, int* sizes, int dim);

void increment(float* arr, int* sizes, int dim, float* output);
tensor_result_t increment2(float* arr, int* sizes, int dim);
tensor_result_t increment3(tensor_result_t arr);

// void convolve(
//     float* input,
//     int* input_sizes,
//     int input_dim,
//     float* kernel,
//     int* kernel_sizes,
//     int kernel_dim,
//     float* output,
//     int* output_sizes,
//     int output_dim
// );

#ifdef __cplusplus
}
#endif

#endif // BRIDGE_H
//hello