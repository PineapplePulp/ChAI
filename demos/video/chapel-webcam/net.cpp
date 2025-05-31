#include "transformer_net.hpp"

int main() {
  torch::manual_seed(0);
  TransformerNet model;
  model->eval();

  // dummy input
  auto input = torch::randn({1,3,256,256});
  auto output = model->forward(input);

  std::cout << "Output shape: " << output.sizes() << "\n";
}