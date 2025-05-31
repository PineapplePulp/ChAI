#include "transformer_net.hpp"





int main() {
  torch::manual_seed(0);
  std::cout << "set seed\n";
  TransformerNet model;
  std::cout << "TransformerNet model created.\n";

  model->load_parameters("/Users/iainmoncrief/Documents/Github/ChAI/demos/video/cpp-model-construction/state_dict_raw.pt");
  // torch::serialize::InputArchive archive;
  // archive.load_from("/Users/iainmoncrief/Documents/Github/ChAI/demos/video/cpp-model-construction/incomplete_sunday_afternoon.model");   // load the raw weights :contentReference[oaicite:2]{index=2}
  // std::cout << "Loading model from archive...\n";
  // model->load(archive);
  // std::cout << "Model loaded successfully!\n";
  // model->eval();
  std::cout << "Model is in evaluation mode.\n";

  // dummy input
  auto input = torch::randn({1,3,256,256});
  std::cout << "Input shape: " << input.sizes() << "\n";
  auto output = model->forward(input);

  std::cout << "Output shape: " << output.sizes() << "\n";
}