#include <torch/torch.h>
#include <torch/script.h>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>

torch::jit::Module load_model(const std::string& model_path) {
    std::cout << "Loading model from path: " << model_path << std::endl;
    torch::jit::Module module;
    try {
        // Deserialize the ScriptModule from a file using torch::jit::load().
        module = torch::jit::load(model_path);
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
    std::string model_path = "style-transfer/models/my_module.pt";
    torch::jit::Module module = load_model(model_path);

    // Create a random input tensor
    torch::Tensor input = torch::randn({10});

    // Run the model
    torch::Tensor output = run_model(module, input);

    // Print the output tensor
    std::cout << "Output tensor: " << output.sizes() << std::endl;

    return 0;
}