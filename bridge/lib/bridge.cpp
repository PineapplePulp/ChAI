#include <bridge.h>

#include <torch/torch.h>
#include <torch/script.h>

// #include <torch/script.h>
// #include <Aten/ATen.h>
#include <iostream>
#include <fstream>
#include <string>
#include <cstring>
#include <sstream>
#include <cstdlib>
#include <vector>
#include <cstdint>




int bridge_tensor_elements(bridge_tensor_t &bt) {
    int size = 1;
    for (int i = 0; i < bt.dim; ++i) {
        size *= bt.sizes[i];
    }
    return size;
}

size_t bridge_tensor_size(bridge_tensor_t &bt) {
    return sizeof(float32_t) * bridge_tensor_elements(bt);
}

void store_tensor(torch::Tensor &input, float32_t* dest) {
    float32_t * data = input.data_ptr<float32_t>();
    size_t bytes_size = sizeof(float32_t) * input.numel();
    std::memmove(dest,data,bytes_size);
}

bridge_tensor_t torch_to_bridge(torch::Tensor &tensor) {
    bridge_tensor_t result;
    result.created_by_c = true;
    result.dim = tensor.dim();
    result.sizes = new int[result.dim];
    for (int i = 0; i < result.dim; ++i) {
        result.sizes[i] = tensor.size(i);
    }
    result.data = new float32_t[bridge_tensor_elements(result)];
    store_tensor(tensor, result.data);
    return result;
}

torch::Tensor bridge_to_torch(bridge_tensor_t &bt) {
    std::vector<int64_t> sizes_vec(bt.sizes, bt.sizes + bt.dim);
    auto shape = at::IntArrayRef(sizes_vec);
    return torch::from_blob(bt.data, shape, torch::kFloat);
}













extern "C" float32_t* unsafe(const float32_t* arr) {
    return const_cast<float32_t*>(arr);
}

std::vector<char> get_the_bytes(std::string filename) {
    std::ifstream input(filename, std::ios::binary);
    std::vector<char> bytes((std::istreambuf_iterator<char>(input)),(std::istreambuf_iterator<char>()));
    input.close();
    return bytes;
}

extern "C" bridge_tensor_t load_tensor_from_file(const uint8_t* file_path) {
    // // Load the tensor from a file
    // torch::Tensor tensor;
    // torch::load(tensor,file_path);

    // std::cout << "Tensor loaded from file: " << tensor.sizes() << std::endl;

    std::string fp(reinterpret_cast<const char*>(file_path));
    std::vector<char> f = get_the_bytes(fp);
    torch::IValue x = torch::pickle_load(f);
    torch::Tensor t = x.toTensor();
    return torch_to_bridge(t);
}

extern "C" bridge_tensor_t load_tensor_dict_from_file(const uint8_t* file_path,const uint8_t* tensor_key) {
    std::string fp(reinterpret_cast<const char*>(file_path));
    std::string tk(reinterpret_cast<const char*>(tensor_key));

    torch::jit::script::Module container = torch::jit::load(fp);
    torch::Tensor tensor = container.attr(tk).toTensor();

    return torch_to_bridge(tensor);

}

extern "C" bridge_tensor_t load_run_model(const uint8_t* model_path, bridge_tensor_t input) {
    auto t_input = bridge_to_torch(input);
    std::string mp(reinterpret_cast<const char*>(model_path));

    std::cout << "Loading model from path: " << mp << std::endl;
    std::cout.flush();


    torch::jit::Module module;
    try
    {
        // Deserialize the ScriptModule from a file using torch::jit::load().
        module = torch::jit::load(mp);
    }
    catch (const c10::Error& e)
    {
        std::cerr << "error loading the model\n" << e.msg();
        std::system("pause");
    }

    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(t_input);

    auto output = module.forward(inputs).toTensor();

    std::cout << "Model output: " << output.sizes() << std::endl;
    return torch_to_bridge(output);
}










extern "C" bridge_tensor_t increment3(bridge_tensor_t arr) {
    auto t = bridge_to_torch(arr);
    // Increment the tensor
    auto incremented_tensor = t + 1;

    return torch_to_bridge(incremented_tensor);
}

extern "C" bridge_tensor_t convolve2d(
    bridge_tensor_t input,
    bridge_tensor_t kernel,
    bridge_tensor_t bias,
    int stride,
    int padding
) {
    auto t_input = bridge_to_torch(input);
    auto t_kernel = bridge_to_torch(kernel);
    auto t_bias = bridge_to_torch(bias);
    auto output = torch::conv2d(t_input, t_kernel, t_bias, stride, padding);
    return torch_to_bridge(output);
}

extern "C" bridge_tensor_t conv2d(
    bridge_tensor_t input,
    bridge_tensor_t kernel,
    bridge_tensor_t bias,
    int stride,
    int padding
) {
    auto t_input = bridge_to_torch(input);
    auto t_kernel = bridge_to_torch(kernel);
    auto t_bias = bridge_to_torch(bias);
    auto output = torch::conv2d(t_input, t_kernel, t_bias, stride, padding);
    return torch_to_bridge(output);
}

extern "C" bridge_tensor_t matmul(bridge_tensor_t a, bridge_tensor_t b) {
    auto t_a = bridge_to_torch(a);
    auto t_b = bridge_to_torch(b);
    auto output = torch::matmul(t_a, t_b);

    // std::cout << "Input A shape: " << t_a.sizes() << std::endl;
    // std::cout << "Input B shape: " << t_b.sizes() << std::endl;
    // std::cout << "Input A: " << t_a.sum() << std::endl;
    // std::cout << "Input B: " << t_b.sum() << std::endl;
    // // std::cout << "Input B: " << t_b << std::endl;
    // std::cout << "Output shape: " << output.sizes() << std::endl;
    // std::cout << "Output sum: " << output.sum() << std::endl;
    // std::cout.flush();
    // printf("Hello from matmul!\n");

    return torch_to_bridge(output);
}

extern "C" bridge_tensor_t max_pool2d(
    bridge_tensor_t input,
    int kernel_size,
    int stride,
    int padding,
    int dilation
) {
    auto t_input = bridge_to_torch(input);
    auto output = torch::max_pool2d(t_input, kernel_size, stride, padding);
    return torch_to_bridge(output);
}

/*

 * Resize the last two dimensions of a tensor, mimicking torchvision.transforms.Resize.
 *
 *  • Works with 2‑D (H,W), 3‑D (C,H,W or N,H,W), 4‑D (N,C,H,W) … tensors – any rank ≥ 2.  
 *  • Leading dimensions are preserved; only the final H × W are resized.  
 *  • Defaults match torchvision: bilinear for floating tensors, align_corners=false.  
 *
 * @param input          Tensor on CPU or CUDA.
 * @param new_h          Target height.
 * @param new_w          Target width.
 * @param mode           Interpolation mode (“bilinear”, “nearest”, “bicubic”, …).
 * @param align_corners  Forwarded to F::interpolate (ignored for “nearest”).
 * @return               Tensor with shape …, new_h, new_w and same dtype / device.

inline torch::Tensor resize_tensor_last2(
        const torch::Tensor& input,
        int64_t              new_h,
        int64_t              new_w,
        const std::string&   mode           = "bilinear",
        bool                 align_corners  = false) {

    // Keep dtype/device; cast to float if interpolate needs it
    const bool need_cast = !input.is_floating_point() && mode != "nearest";
    auto x = need_cast ? input.to(torch::kFloat32) : input;
    x = x.contiguous();                         // guarantees a re‑view is safe

    // Collapse every axis except the last two into a single batch dimension.
    const int64_t h  = x.size(-2);
    const int64_t w  = x.size(-1);
    const int64_t flat = x.numel() / (h * w);   // product of leading dims

    auto x4d = x.view({flat, 1, h, w});         // N=flat, C=1, H, W

    // Interpolate – equivalent to torchvision.transforms.Resize for tensors.
    auto y4d = torch::nn::functional::interpolate(
        x4d,
        torch::nn::functional::InterpolateFuncOptions()
            .size(std::vector<int64_t>{new_h, new_w})
            .mode(mode)
            .align_corners(align_corners));

    // Restore the original leading shape.
    std::vector<int64_t> out_shape(input.sizes().begin(), input.sizes().end() - 2);
    out_shape.push_back(new_h);
    out_shape.push_back(new_w);

    auto y = y4d.view(out_shape);
    return need_cast ? y.to(input.scalar_type()) : y;
}
*/
extern "C" bridge_tensor_t resize(
    bridge_tensor_t input,
    int height,
    int width
) {
    auto image = bridge_to_torch(input);

    // auto output = resize_tensor_last2(image, height, width);
    
    // at::Tensor output = at::upsample_bilinear2d(t_input.unsqueeze(0), {height, width}, false);
    if (image.dim() == 3) {
        auto output = torch::nn::functional::interpolate(
            image.unsqueeze(0),
            torch::nn::functional::InterpolateFuncOptions()
            .size(std::vector<int64_t>({ height, width }))
            .mode(torch::kBilinear)
            .align_corners(false)
        ).squeeze(0);
        return torch_to_bridge(output);
    } else if (image.dim() == 4) {
        auto output = torch::nn::functional::interpolate(
            image,
            torch::nn::functional::InterpolateFuncOptions()
            .size(std::vector<int64_t>({ height, width }))
            .mode(torch::kBilinear)
            .align_corners(false)
        );
        return torch_to_bridge(output);
    } else {
        std::cerr << "Unsupported tensor dimension: " << image.dim() << std::endl;
        std::cerr.flush();
        std::cout << "Unsupported tensor dimension: " << image.dim() << std::endl;
        std::cout.flush();
        return input; // Return the original tensor if the dimension is unsupported
    }
}


// extern "C"


//  
// extern "C" bridge_tensor_t conv2d(
//     bridge_tensor_t input,
//     bridge_tensor_t kernel,
//     nil_scalar_tensor_t bias,
//     nil_scalar_tensor_t stride,
//     nil_scalar_tensor_t padding
// ) {
//     namespace F = torch::nn::functional;
//     F::conv2d(input, kernel, F::Conv2dFuncOptions().stride(1));
// }


extern "C" int baz(void) {
    printf("Hello from baz!\n");
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

extern "C" bridge_tensor_t increment2(float* arr, int* sizes, int dim) {
    // Convert sizes to std::vector<int64_t>
    std::vector<int64_t> sizes_vec(sizes, sizes + dim);
    auto shape = at::IntArrayRef(sizes_vec);
    auto t = torch::from_blob(arr, shape, torch::kFloat);

    // // Increment the tensor
    // auto incremented_tensor = t + 1;

    // // Store the incremented tensor in the output array
    // storeTensor(incremented_tensor, output);

    auto incremented_tensor = t + 1;

    return torch_to_bridge(incremented_tensor);
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