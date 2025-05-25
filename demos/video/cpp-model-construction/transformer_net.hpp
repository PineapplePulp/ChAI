// transformer_net.h
#pragma once
#include <torch/torch.h>
#include <string>
#include <iostream>
#include <fstream>
#include <vector>
//
// --- ConvLayer -------------------------------------------------------------
//
struct ConvLayerImpl : torch::nn::Module {
  torch::nn::ReflectionPad2d reflection_pad{nullptr};
  torch::nn::Conv2d            conv2d{nullptr};

  ConvLayerImpl(int64_t in_channels,
                int64_t out_channels,
                int64_t kernel_size,
                int64_t stride)
    : reflection_pad(torch::nn::ReflectionPad2dOptions(kernel_size / 2)),
      conv2d(torch::nn::Conv2dOptions(in_channels, out_channels, kernel_size)
               .stride(stride)) 
  {
    register_module("reflection_pad", reflection_pad);
    register_module("conv2d",        conv2d);
  }

  torch::Tensor forward(torch::Tensor x) {
    x = reflection_pad->forward(x);
    x = conv2d->forward(x);
    return x;
  }
};
TORCH_MODULE(ConvLayer);

//
// --- ResidualBlock ---------------------------------------------------------
//
struct ResidualBlockImpl : torch::nn::Module {
  ConvLayer                   conv1{nullptr};
  torch::nn::InstanceNorm2d   in1{nullptr};
  ConvLayer                   conv2{nullptr};
  torch::nn::InstanceNorm2d   in2{nullptr};
  torch::nn::ReLU             relu{nullptr};

  ResidualBlockImpl(int64_t channels)
    : conv1(ConvLayer(channels, channels, 3, 1)),
      in1(torch::nn::InstanceNorm2dOptions(channels).affine(true)),
      conv2(ConvLayer(channels, channels, 3, 1)),
      in2(torch::nn::InstanceNorm2dOptions(channels).affine(true)),
      relu(torch::nn::ReLUOptions(true))
  {
    register_module("conv1", conv1);
    register_module("in1",   in1);
    register_module("conv2", conv2);
    register_module("in2",   in2);
    register_module("relu",  relu);
  }

  torch::Tensor forward(torch::Tensor x) {
    auto residual = x;
    auto out = relu->forward(in1->forward(conv1->forward(x)));
    out = in2->forward(conv2->forward(out));
    return out + residual;
  }
};
TORCH_MODULE(ResidualBlock);

//
// --- UpsampleConvLayer -----------------------------------------------------
//
struct UpsampleConvLayerImpl : torch::nn::Module {
  torch::nn::Upsample         upsample{nullptr};
  torch::nn::ReflectionPad2d  reflection_pad{nullptr};
  torch::nn::Conv2d           conv2d{nullptr};

  UpsampleConvLayerImpl(int64_t in_channels,
                        int64_t out_channels,
                        int64_t kernel_size,
                        int64_t stride,
                        int64_t upsample_scale)
    : upsample(torch::nn::UpsampleOptions()
                   .scale_factor(std::vector<double>{(double)upsample_scale, (double)upsample_scale})
                   .mode(torch::kNearest)),
      reflection_pad(torch::nn::ReflectionPad2dOptions(kernel_size / 2)),
      conv2d(torch::nn::Conv2dOptions(in_channels, out_channels, kernel_size)
               .stride(stride))
  {
    register_module("upsample",        upsample);
    register_module("reflection_pad",  reflection_pad);
    register_module("conv2d",          conv2d);
  }

  torch::Tensor forward(torch::Tensor x) {
    x = upsample->forward(x);
    x = reflection_pad->forward(x);
    x = conv2d->forward(x);
    return x;
  }
};
TORCH_MODULE(UpsampleConvLayer);

//
// --- TransformerNet --------------------------------------------------------
//
struct TransformerNetImpl : torch::nn::Module {
  ConvLayer                   conv1{nullptr};
  torch::nn::InstanceNorm2d   in1{nullptr};
  ConvLayer                   conv2{nullptr};
  torch::nn::InstanceNorm2d   in2{nullptr};
  ConvLayer                   conv3{nullptr};
  torch::nn::InstanceNorm2d   in3{nullptr};
  ResidualBlock               res1{nullptr}, res2{nullptr}, res3{nullptr},
                              res4{nullptr}, res5{nullptr};
  UpsampleConvLayer           deconv1{nullptr}, deconv2{nullptr};
  torch::nn::InstanceNorm2d   in4{nullptr}, in5{nullptr};
  ConvLayer                   deconv3{nullptr};
  torch::nn::ReLU             relu{nullptr};

  TransformerNetImpl()
    : conv1(ConvLayer(  3,  32, 9, 1)),
      in1  (torch::nn::InstanceNorm2dOptions(32).affine(true)),
      conv2(ConvLayer( 32,  64, 3, 2)),
      in2  (torch::nn::InstanceNorm2dOptions(64).affine(true)),
      conv3(ConvLayer( 64, 128, 3, 2)),
      in3  (torch::nn::InstanceNorm2dOptions(128).affine(true)),
      res1 (ResidualBlock(128)),
      res2 (ResidualBlock(128)),
      res3 (ResidualBlock(128)),
      res4 (ResidualBlock(128)),
      res5 (ResidualBlock(128)),
      deconv1(UpsampleConvLayer(128,  64, 3, 1, 2)),
      in4    (torch::nn::InstanceNorm2dOptions(64).affine(true)),
      deconv2(UpsampleConvLayer( 64,  32, 3, 1, 2)),
      in5    (torch::nn::InstanceNorm2dOptions(32).affine(true)),
      deconv3(ConvLayer(        32,   3, 9, 1)),
      relu   (torch::nn::ReLUOptions(true))
  {
    register_module("conv1",   conv1);
    register_module("in1",     in1);
    register_module("conv2",   conv2);
    register_module("in2",     in2);
    register_module("conv3",   conv3);
    register_module("in3",     in3);
    register_module("res1",    res1);
    register_module("res2",    res2);
    register_module("res3",    res3);
    register_module("res4",    res4);
    register_module("res5",    res5);
    register_module("deconv1", deconv1);
    register_module("in4",     in4);
    register_module("deconv2", deconv2);
    register_module("in5",     in5);
    register_module("deconv3", deconv3);
    register_module("relu",    relu);
  }

  torch::Tensor forward(torch::Tensor x) {
    x = relu->forward(in1->forward(conv1->forward(x)));
    x = relu->forward(in2->forward(conv2->forward(x)));
    x = relu->forward(in3->forward(conv3->forward(x)));
    x = res1->forward(x);
    x = res2->forward(x);
    x = res3->forward(x);
    x = res4->forward(x);
    x = res5->forward(x);
    x = relu->forward(in4->forward(deconv1->forward(x)));
    x = relu->forward(in5->forward(deconv2->forward(x)));
    x = deconv3->forward(x);
    return x;
  }

    // Model class is inherited from public nn::Module
  std::vector<char> get_the_bytes(std::string filename) {
      std::ifstream input(filename, std::ios::binary);
      std::vector<char> bytes(
          (std::istreambuf_iterator<char>(input)),
          (std::istreambuf_iterator<char>()));

      input.close();
      return bytes;
  }

  void load_parameters(std::string pt_pth) {
    std::vector<char> f = this->get_the_bytes(pt_pth);
    c10::Dict<at::IValue, at::IValue> weights = torch::pickle_load(f).toGenericDict();

    const torch::OrderedDict<std::string, at::Tensor>& model_params = this->named_parameters();
    std::vector<std::string> param_names;
    for (auto const& w : model_params) {
      param_names.push_back(w.key());
    }

    torch::NoGradGuard no_grad;
    for (auto const& w : weights) {
        std::string name = w.key().toStringRef();
        at::Tensor param = w.value().toTensor();

        if (std::find(param_names.begin(), param_names.end(), name) != param_names.end()){
          model_params.find(name)->copy_(param);
        } else {
          std::cout << name << " does not exist among model parameters." << std::endl;
        };

    }
  }
};
TORCH_MODULE(TransformerNet);
