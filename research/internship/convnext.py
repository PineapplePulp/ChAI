import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm2d(nn.Module):
    def __init__(self, in_channels):
      super().__init__()
      self.layer_norm = nn.LayerNorm(in_channels)
      # self.weight = nn.Parameter(torch.ones(1, in_channels, 1, 1))
      # self.bias = nn.Parameter(torch.zeros(1, in_channels, 1, 1))

    def forward(self, x: Tensor) -> Tensor:
      x = torch.permute(x, (0, 2, 3, 1))
      x = self.layer_norm(x)
      x = torch.permute(x, (0, 3, 1, 2))
      return x

class ConvNextStem(nn.Module):
   def __init__(self, in_channels, out_channels, kernel_size=3):
    super().__init__()
    self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=kernel_size)
    self.layer_norm_2d = LayerNorm2d(out_channels)

   def forward(self,x):
    x = self.conv(x)
    x = self.layer_norm_2d(x)
    return x


class ConvNextBlock(nn.Module):

  def __init__(self, d_in, layer_scale=1e-6, kernel_size=7, stochastic_depth_prob=1):
    super().__init__()
    self.stochastic_depth_prob = stochastic_depth_prob
    self.depthwise_conv = nn.Conv2d(d_in, d_in, kernel_size=7, padding='same', groups=d_in)
    self.layer_norm_2d = LayerNorm2d(d_in)
    self.conv1 = nn.Conv2d(d_in, 4*d_in, kernel_size=1)
    self.gelu = nn.GELU()
    self.conv2 = nn.Conv2d(4*d_in, d_in, kernel_size=1)
    self.layer_scale = nn.Parameter(layer_scale*torch.ones(1, d_in, 1, 1))

  def forward(self,x):
    residual = x
    x = self.depthwise_conv(x)
    x = self.layer_norm_2d(x)
    x = self.conv1(x)
    x = self.gelu(x)
    x = self.conv2(x)
    x = self.layer_scale * x

    if self.training:
      if torch.rand(1).item() > self.stochastic_depth_prob:
        return residual
      else: 
        return residual + x

    x = residual + self.stochastic_depth_prob * x
    return x
    

class ConvNextDownsample(nn.Module):
  def __init__(self, d_in, d_out, width=2):
    super().__init__()
    self.layer_norm = LayerNorm2d(d_in)
    self.conv = nn.Conv2d(d_in, d_out, kernel_size=width, stride=width)

  def forward(self,x):
    x = self.layer_norm(x)
    x = self.conv(x)
    return x
    

class ConvNextClassifier(nn.Module):
  def __init__(self, d_in, d_out):
    super().__init__()
    self.global_pool = nn.AdaptiveAvgPool2d((1,1))
    self.flatten = nn.Flatten()
    self.layer_norm = nn.LayerNorm(d_in)
    self.linear = nn.Linear(d_in, d_out)

  def forward(self,x):
    x = self.global_pool(x)
    x = self.flatten(x)
    x = self.layer_norm(x)
    x = self.linear(x)
    return x


class ConvNext(nn.Module):

  def __init__(self, in_channels, out_channels, blocks=[96]):
    super().__init__()
    num_blocks = len(blocks)
    self.layers = nn.ModuleList()
    self.layers.append(ConvNextStem(in_channels=in_channels, out_channels=blocks[0]))

    for i in range(num_blocks):
      # stochastic_depth_prob = 1-(i/num_blocks)*0.5
      stochastic_depth_prob = 0.001
      self.layers.append(ConvNextBlock(d_in=blocks[i], stochastic_depth_prob=stochastic_depth_prob))

      if i+1 < num_blocks and blocks[i] != blocks[i+1]:
        self.layers.append(ConvNextDownsample(d_in=blocks[i], d_out=blocks[i+1]))

    self.layers.append(ConvNextClassifier(d_in=blocks[-1], d_out=out_channels))

    for module in self.modules():
      if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
        nn.init.trunc_normal_(module.weight, mean=0, std=0.02, a=-2, b=2)
        if module.bias is not None:
          nn.init.zeros_(module.bias)

      elif isinstance(module, nn.LayerNorm):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)

  def forward(self,x):
    for layer in self.layers:
      x = layer(x)
    return x