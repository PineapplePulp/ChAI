#include <torch/torch.h>
#include <iostream>


int main() {
    torch::Device device(torch::kMPS);

    torch::Tensor a = torch::rand({2, 3}, device);
    std::cout << a << std::endl;
    std::cout << "Tensor: " << a.device() << std::endl;

    torch::Tensor b = torch::rand({2, 3});
    std::cout << b << std::endl;
    std::cout << "Tensor: " << b.device() << std::endl;
    return 0;
}