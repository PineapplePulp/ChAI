#include <bridge.h>

#include <torch/torch.h>
// #include <torch/script.h>
// #include <Aten/ATen.h>
#include <iostream>
#include <vector>

void secret() {
    std::cout << "Secret function called!" << std::endl;
    torch::Tensor tensor = torch::eye(3);
    // std::cout << "Tensor: " << tensor << std::endl;

    // std::cout << "Secret function called! Tensor: " << tensor << std::endl;
}

extern "C" int baz(void) {
    printf("Hello from baz!\n");
    secret();
    // auto x = torch::randn({5, 3});
    // return x.size(0);
}


extern "C" void wrHello(void) {
    printf("Hello from wrHello!\n");
}


extern "C" void wrHelloTorch(void) {
    printf("Hello from wrHelloTorch!\n");
    // auto t = torch::ones({2, 3});
    // std::cout << t << std::endl;
}

extern "C" float sumArray(float* arr, int* sizes, int dim) {
    // Convert sizes to std::vector<int64_t>

    printf("sumArray called with arr: %p, sizes: %p, dim: %d\n", arr, sizes, dim);

    // std::vector<int64_t> sizes_vec(sizes, sizes + dim);
    // std::cout << sizes_vec << std::endl;

    // auto shape = at::IntArrayRef(sizes_vec);
    // std::cout << shape << std::endl;

    // auto t = torch::from_blob(arr, shape, torch::kFloat);
    // std::cout << t << std::endl;

    // return t.sum().item<float>();

    return 0.0f;

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