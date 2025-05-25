#ifndef BRIDGE_H
#define BRIDGE_H

#include <stdio.h>

#ifndef __cplusplus
#include <stdbool.h>
#endif

#ifdef __cplusplus
#include <cstdint>
#else
#include <stdint.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define proto_bridge_simple(Name) \
    bridge_tensor_t Name(bridge_tensor_t input)

typedef float float32_t;
typedef double float64_t;
typedef char bool_t;
typedef unsigned char uint8_t;
typedef unsigned int uint32_t;
typedef unsigned long long uint64_t;

void debug_cpu_only_mode(bool_t mode);

typedef struct bridge_tensor_t {
    float* data;
    int* sizes;
    int dim;
    bool_t created_by_c;
} bridge_tensor_t;


typedef struct bridge_pt_model_t {
    void* pt_module;
} bridge_pt_model_t;

typedef struct test_struct_t {
    int* field;
} test_struct_t;


void hello_world(void);

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

bridge_pt_model_t load_model(const uint8_t* model_path);

bool_t accelerator_available(void);

bridge_tensor_t model_forward(bridge_pt_model_t model, bridge_tensor_t input);
bridge_tensor_t model_forward_style_transfer(bridge_pt_model_t model, bridge_tensor_t input);

bridge_tensor_t resize(bridge_tensor_t input,int height,int width);
bridge_tensor_t imagenet_normalize(bridge_tensor_t input);


bridge_tensor_t add_two_arrays(bridge_tensor_t a, bridge_tensor_t b);

// bridge_tensor_t capture_webcam_bridge(int cam_index);

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

void split_loop(int64_t idx, int64_t n);
void split_loop_filler(int64_t n,int64_t* ret);


// bridge_tensor_t conv2d(
//     bridge_tensor_t input,
//     bridge_tensor_t kernel,
//     nil_scalar_tensor_t bias,
//     nil_scalar_tensor_t stride,
//     nil_scalar_tensor_t padding
// );

proto_bridge_simple(gelu);

proto_bridge_simple(logsigmoid);

proto_bridge_simple(mish);

proto_bridge_simple(relu);

proto_bridge_simple(relu6);

proto_bridge_simple(selu);

proto_bridge_simple(silu);

proto_bridge_simple(softsign);

proto_bridge_simple(tanhshrink);


bridge_tensor_t rrelu(bridge_tensor_t input, float lower, float upper, bool training);

bridge_tensor_t hardshrink(bridge_tensor_t input, float lambda);

bridge_tensor_t hardtanh(bridge_tensor_t input, float min_val, float max_val);

bridge_tensor_t elu(bridge_tensor_t input, float alpha);

bridge_tensor_t softplus(bridge_tensor_t input, float beta, float threhsold);

bridge_tensor_t threshold(bridge_tensor_t input, float threshold, float value);

bridge_tensor_t celu(bridge_tensor_t input, float alpha);

bridge_tensor_t leaky_relu(bridge_tensor_t input, float negative_slope);

bridge_tensor_t softshrink(bridge_tensor_t input, float lambda);

#ifdef __cplusplus
bridge_tensor_t softmax(bridge_tensor_t input, std::int64_t dim);
#else
bridge_tensor_t softmax(bridge_tensor_t input, int64_t dim);
#endif

#ifdef __cplusplus
bridge_tensor_t softmin(bridge_tensor_t input, std::int64_t dim);
#else
bridge_tensor_t softmin(bridge_tensor_t input, int64_t dim);
#endif

bridge_tensor_t dropout(bridge_tensor_t input, double p, bool training);

bridge_tensor_t alpha_dropout(bridge_tensor_t input, double p, bool training);

bridge_tensor_t feature_alpha_dropout(bridge_tensor_t input, double p, bool training);

bridge_tensor_t dropout2d(bridge_tensor_t input, double p, bool training);

bridge_tensor_t dropout3d(bridge_tensor_t input, double p, bool training);

#undef proto_bridge_simple

#ifdef __cplusplus
}
#endif

#endif // BRIDGE_H