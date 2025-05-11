#include <torch/torch.h>

#include <iostream>



torch::Tensor sobel_rgb(const torch::Tensor& img)
{
    TORCH_CHECK(img.dim() == 4 &&
                img.size(0) == 1 &&
                img.size(1) == 3,
                "Expected input of shape [1, 3, H, W]");

    // Preserve dtype & device
    const auto opts = img.options();
    const auto dev  = img.device();

    /* ----- build 3-channel Sobel kernels -------------------------------- */
    // (out_channels, in_channels / groups, kH, kW) with groups = 3
    at::Tensor kx = torch::tensor({{ -1,  0,  1},
                                    { -2,  0,  2},
                                    { -1,  0,  1}}, opts);
    at::Tensor ky = torch::tensor({{ -1, -2, -1},
                                    {  0,  0,  0},
                                    {  1,  2,  1}}, opts);

    // Replicate each kernel for the three groups (RGB)
    at::Tensor weight_x = kx.expand({3, 1, 3, 3}).clone();
    at::Tensor weight_y = ky.expand({3, 1, 3, 3}).clone();

    /* ----- convolutions -------------------------------------------------- */
    const int64_t groups = 3;
    const int64_t padding = 1;  // keep spatial size

    at::Tensor gx = torch::conv2d(img, weight_x, /*bias=*/{}, /*stride=*/1,
                                    padding, /*dilation=*/1, groups);
    at::Tensor gy = torch::conv2d(img, weight_y, /*bias=*/{}, /*stride=*/1,
                                    padding, /*dilation=*/1, groups);

    /* ----- gradient magnitude ------------------------------------------- */
    at::Tensor magnitude = torch::sqrt(gx.pow(2) + gy.pow(2) + 1e-12);

    return magnitude;
}



int main() {

    at::Tensor input = torch::rand({1, 3, 10, 10}).to(torch::kFloat32);
    std::cout << "Input: " << input.sizes() << std::endl;

    auto output = sobel_rgb(input);
    
    std::cout << "Output: " << output.sizes() << std::endl;

    return 0;
}