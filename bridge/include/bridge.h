#ifndef BRIDGE_H
#define BRIDGE_H

#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef float float32_t;
typedef double float64_t;
typedef char bool_t;
typedef unsigned char uint8_t;
typedef unsigned int uint32_t;

typedef struct bridge_tensor_t {
    float* data;
    int* sizes;
    int dim;
    bool_t created_by_c;
} bridge_tensor_t;


typedef struct nil_scalar_tensor_t {
    float scalar;
    bridge_tensor_t tensor;
    bool_t is_nil;
    bool_t is_scalar;
    bool_t is_tensor;
} nil_scalar_tensor_t;

float* unsafe(const float* arr);
bridge_tensor_t load_tensor_from_file(const uint8_t* file_path);
bridge_tensor_t load_tensor_dict_from_file(const uint8_t* file_path,const uint8_t* tensor_key);
bridge_tensor_t load_run_model(const uint8_t* model_path, bridge_tensor_t input);


int baz(void);

void wrHello(void);

void wrHelloTorch(void);

float sumArray(float* arr, int* sizes, int dim);

void increment(float* arr, int* sizes, int dim, float* output);
bridge_tensor_t increment2(float* arr, int* sizes, int dim);
bridge_tensor_t increment3(bridge_tensor_t arr);

bridge_tensor_t convolve2d(
    bridge_tensor_t input,
    bridge_tensor_t kernel,
    bridge_tensor_t bias,
    int stride,
    int padding
);

bridge_tensor_t conv2d(
    bridge_tensor_t input,
    bridge_tensor_t kernel,
    bridge_tensor_t bias,
    int stride,
    int padding
);

bridge_tensor_t matmul(bridge_tensor_t a, bridge_tensor_t b);

bridge_tensor_t max_pool2d(
    bridge_tensor_t input,
    int kernel_size,
    int stride,
    int padding,
    int dilation
);


// bridge_tensor_t conv2d(
//     bridge_tensor_t input,
//     bridge_tensor_t kernel,
//     nil_scalar_tensor_t bias,
//     nil_scalar_tensor_t stride,
//     nil_scalar_tensor_t padding
// );



#ifdef __cplusplus
}
#endif

#endif // BRIDGE_H
//hello