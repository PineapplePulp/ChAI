#include <torch/torch.h>
#include <torch/script.h>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>

torch::Device default_device = torch::Device(torch::kCPU);


torch::jit::Module load_model(const std::string& model_path) {
    std::cout << "Loading model from path: " << model_path << std::endl;
    torch::jit::Module module;
    try {
        // Deserialize the ScriptModule from a file using torch::jit::load().
        module = torch::jit::load(model_path);
        std::cout << "Model loaded successfully." << std::endl;

        std::cout << "Moving model to device..." << std::endl;
        module.to(default_device);
        std::cout << "Model moved to device." << std::endl;

    } catch (const c10::Error& e) {
        std::cerr << "error loading the model\n" << e.msg();
    }
    std::cout << "Model loaded successfully." << std::endl;
    return module;

}

torch::Tensor run_model(torch::jit::Module& module, const torch::Tensor& input) {
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(input);

    std::cout << "Input tensor: " << input.sizes() << std::endl;
    auto output = module.forward(inputs).toTensor();
    std::cout << "Model output: " << output.sizes() << std::endl;
    return output;
}


int main() {
    // Load the model
    // std::string model_path = "style-transfer/models/my_module.pt";
    // torch::jit::Module module = load_model(model_path);
    // torch::Tensor input = torch::randn({10});
    // torch::Tensor output = run_model(module, input);


    if (torch::mps::is_available()) {
        default_device = torch::Device(torch::kMPS);
        std::cout << "MPS is available and set as the default device." << std::endl;
    } else {
        default_device = torch::Device(torch::kCPU);
        std::cout << "MPS is not available. Using CPU instead." << std::endl;
    }

    std::string model_path = "style-transfer/models/mosaic.pt";
    torch::jit::Module module = load_model(model_path);
    torch::Tensor input = torch::randn({1, 3, 1428, 1904}, default_device);
    torch::Tensor output = run_model(module, input);


    // Print the output tensor
    std::cout << "Output tensor: " << output.sizes() << std::endl;

    return 0;
}