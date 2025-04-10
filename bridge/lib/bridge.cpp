#include <bridge.h>

#include <torch/torch.h>
// #include <torch/script.h>
// #include <Aten/ATen.h>
#include <iostream>
#include <vector>

#include <cstdint>

using float32_t = float;


int tensor_result_elements(tensor_result_t &result) {
    int size = 1;
    for (int i = 0; i < result.dim; ++i) {
        size *= result.sizes[i];
    }
    return size;
}

size_t tensor_result_size(tensor_result_t &result) {
    return sizeof(float32_t) * tensor_result_elements(result);
}

void store_tensor(torch::Tensor &input, float32_t* dest) {
    float32_t * data = input.data_ptr<float32_t>();
    size_t bytes_size = sizeof(float32_t) * input.numel();
    std::memmove(dest,data,bytes_size);
}

tensor_result_t tensor_result_convert(torch::Tensor &tensor) {
    tensor_result_t result;
    result.dim = tensor.dim();
    result.sizes = new int[result.dim];
    for (int i = 0; i < result.dim; ++i) {
        result.sizes[i] = tensor.size(i);
    }
    result.data = new float32_t[tensor_result_elements(result)];
    store_tensor(tensor, result.data);
    return result;
}

void secret() {
    std::cout << "Secret function called!" << std::endl;
    auto x = torch::randn({5, 3});
    // torch::Tensor tensor = torch::eye(3);
    std::cout << "Tensor: " << x << std::endl;

    // std::cout << "Secret function called! Tensor: " << tensor << std::endl;
}

extern "C" int baz(void) {
    printf("Hello from baz!\n");
    secret();
    auto x = torch::randn({5, 3});
    return x.size(0);
}


extern "C" void wrHello(void) {
    printf("Hello from wrHello!\n");
}


extern "C" void wrHelloTorch(void) {
    printf("Hello from wrHelloTorch!\n");
    // auto t = torch::ones({2, 3});
    // std::cout << t << std::endl;
}




extern "C" void increment(float* arr, int* sizes, int dim, float* output) {
    // Convert sizes to std::vector<int64_t>
    std::vector<int64_t> sizes_vec(sizes, sizes + dim);
    auto shape = at::IntArrayRef(sizes_vec);
    auto t = torch::from_blob(arr, shape, torch::kFloat);

    // // Increment the tensor
    // auto incremented_tensor = t + 1;

    // // Store the incremented tensor in the output array
    // storeTensor(incremented_tensor, output);

    auto incremented_tensor = torch::from_blob(output, shape, torch::kFloat);
    incremented_tensor.copy_(t + 1);
}

extern "C" tensor_result_t increment2(float* arr, int* sizes, int dim) {
    // Convert sizes to std::vector<int64_t>
    std::vector<int64_t> sizes_vec(sizes, sizes + dim);
    auto shape = at::IntArrayRef(sizes_vec);
    auto t = torch::from_blob(arr, shape, torch::kFloat);

    // // Increment the tensor
    // auto incremented_tensor = t + 1;

    // // Store the incremented tensor in the output array
    // storeTensor(incremented_tensor, output);

    auto incremented_tensor = t + 1;

    return tensor_result_convert(incremented_tensor);
}

extern "C" tensor_result_t increment3(tensor_result_t arr) {
    // Convert sizes to std::vector<int64_t>
    std::vector<int64_t> sizes_vec(arr.sizes, arr.sizes + arr.dim);
    auto shape = at::IntArrayRef(sizes_vec);
    auto t = torch::from_blob(arr.data, shape, torch::kFloat);

    // Increment the tensor
    auto incremented_tensor = t + 1;

    return tensor_result_convert(incremented_tensor);
}


extern "C" float sumArray(float* arr, int* sizes, int dim) {
    // Convert sizes to std::vector<int64_t>

    printf("sumArray called with arr: %p, sizes: %p, dim: %d\n", arr, sizes, dim);

    std::vector<int64_t> sizes_vec(sizes, sizes + dim);
    std::cout << sizes_vec << std::endl;

    auto shape = at::IntArrayRef(sizes_vec);
    std::cout << shape << std::endl;

    auto t = torch::from_blob(arr, shape, torch::kFloat);
    std::cout << t << std::endl;

    return t.sum().item<float>();

    // return 0.0f;

    // float sum = 0.0f;
    // for (int i = 0; i < size; ++i) {
    //     sum += arr[i];
    // }
    // return sum;
    // const std::vector<int64_t> sizes_vec(sizes, dim);
    // auto shape = at::IntArrayRef(sizes_vec);

    // auto t = torch::from_blob(arr, shape, torch::kFloat);
    // return t.sum().item<float>();
}