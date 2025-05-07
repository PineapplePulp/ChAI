import cv2
import torch
import numpy as np
import utils
import torchvision
import argparse



# default_device = torch.device('cpu')
# if torch.backends.mps.is_available():
#     default_device = torch.device('mps')
#     print('using mps')

# if torch.backends.cuda.is_available():
#     default_device = torch.device('cuda')
#     print('using cuda')

# print('using default device:', default_device)

# torch.set_default_device(default_device)



# Open the default camera
cam = cv2.VideoCapture(0)

# Get the default frame width and height
frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (frame_width, frame_height))


import torch.nn.functional as F


# sobel_dx = torch.tensor([[-1, 0, 1],
#                          [-2, 0, 2],
#                          [-1, 0, 1]], dtype=torch.float32)

# sobel_dy = torch.tensor([[-1, -2, -1],
#                          [ 0,  0,  0],
#                          [ 1,  2,  1]], dtype=torch.float32)

# kernel = torch.cat([sobel_dx.unsqueeze(0), sobel_dy.unsqueeze(0)],0)   # [2,3,3]
# kernel = kernel.unsqueeze(1).to('mps')  # [2,3,3,3]

# # kernel = kernel.unsqueeze(1).repeat(1, 3, 1, 1).to('mps')  # [2,3,3,3]

# def sobel_filter(img: torch.Tensor) -> torch.Tensor:
#     """
#     img: Nx3xHxW float32 in [0,1] or [0,255]
#     returns: Nx2xHxW  (channel 0 = ∂I/∂x, channel 1 = ∂I/∂y)
#     """
#     return F.conv2d(img, kernel, padding=1)

# def sobel_magnitude(img: torch.Tensor) -> torch.Tensor:
#     g = sobel_filter(img)
#     return (g ** 2).sum(1, keepdim=True).sqrt()


def sobel_edges(rgb: torch.Tensor) -> torch.Tensor:
    """
    rgb : (N, 3, H, W) float tensor in the range [0, 1] or [-1, 1]
          (any range is fine as long as it's float)
    
    Returns
    -------
    edges : (N, 3, H, W) tensor – per‑channel Sobel edge magnitude,
            same H and W as the input (no cropping or padding artifacts).
    """
    # --- 1. Build Sobel kernels ------------------------------------------------
    sobel_x = torch.tensor([[-1., 0., 1.],
                            [-2., 0., 2.],
                            [-1., 0., 1.]],requires_grad=False).to('mps')
    sobel_y = sobel_x.T

    # Each colour channel must be convolved with *its own* kernel.
    # We therefore use depth‑wise (grouped) convolution with groups=3.
    # Weight shape for conv2d: (out_channels, in_channels/groups, kH, kW)
    # Here:  out_channels = in_channels = 3   and   groups = 3
    weight_x = sobel_x.expand(3, 1, 3, 3).to(rgb)       # (3,1,3,3)
    weight_y = sobel_y.expand(3, 1, 3, 3).to(rgb)

    # --- 2. Apply the 2D convolutions -----------------------------------------
    # Kernel size is 3 ⇒ one‑pixel border is enough to keep size unchanged.
    grad_x = F.conv2d(rgb, weight_x, padding=1, groups=3)
    grad_y = F.conv2d(rgb, weight_y, padding=1, groups=3)

    # --- 3. Edge magnitude per channel ----------------------------------------
    # A small epsilon avoids a zero‑gradient sqrt warning.
    edges = torch.sqrt(grad_x**2 + grad_y**2 + 1e-6)

    return edges


def tensor_to_bgr(frame_tensor, *, undo_normalise=False, mean=None, std=None):
    """
    Args
    ----
    frame_tensor : torch.Tensor
        (C,H,W) or (1,C,H,W)   ―  float or half   ―  RGB
    undo_normalise : bool
        True if you previously applied (x - mean) / std
    mean, std : list/tuple of 3 floats
        Same numbers you used for normalising (e.g. ImageNet)
    Returns
    -------
    frame_bgr : np.ndarray   (H,W,3) uint8   BGR  contiguous
    """
    # 1) squeeze batch dimension if present
    if frame_tensor.ndim == 4:
        frame_tensor = frame_tensor[0]

    # 2) move to CPU & float32 for math
    img = frame_tensor.detach()

    # 3) (optional) reverse mean/std normalisation
    if undo_normalise:
        if mean is None or std is None:
            raise ValueError("Supply mean and std to undo normalisation")
        mean = torch.tensor(mean).to(img).view(3,1,1)
        std  = torch.tensor(std).to(img).view(3,1,1)
        img = img * std + mean

    # 4) scale back to 0‑255, clamp, uint8
    img = (img * 255.0)
    # img = img # .to(torch.float16)
    img = img.clamp(0,255).byte()

    # 5) channel‑last & numpy
    img = img.permute(1,2,0).cpu().numpy()                 # H,W,C  RGB
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)       # → BGR
    img = np.ascontiguousarray(img)                  # ensure OpenCV‑happy
    return img


def undo_normalize(tensor):
    mean = (0.485, 0.456, 0.406)  # ImageNet defaults (RGB)
    std  = (0.229, 0.224, 0.225)
    mean = torch.tensor(mean, dtype=tensor.dtype, device=tensor.device)[:, None, None]
    std  = torch.tensor(std,  dtype=tensor.dtype, device=tensor.device)[:, None, None]
    return (tensor * std + mean).clamp(0, 1)


class Sobel(torch.nn.Module):
    def __init__(self):
        super(Sobel, self).__init__()

        # self.sobel_kernel = torch.nn.Parameter(sobel_kernel, requires_grad=False)
        # self.sobel_cnn = torch.nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1, bias=False).to(torch.float16)
        # self.sobel_cnn.weight = torch.nn.Parameter(sobel_kernel, requires_grad=False)

    def forward(self, rgb):
        # return self.sobel_cnn(x)
        sobel_x = torch.tensor([[-1., 0., 1.],
                            [-2., 0., 2.],
                            [-1., 0., 1.]],requires_grad=False)
        sobel_y = sobel_x.T

        # Each colour channel must be convolved with *its own* kernel.
        # We therefore use depth‑wise (grouped) convolution with groups=3.
        # Weight shape for conv2d: (out_channels, in_channels/groups, kH, kW)
        # Here:  out_channels = in_channels = 3   and   groups = 3
        weight_x = sobel_x.expand(3, 1, 3, 3).to(rgb)       # (3,1,3,3)
        weight_y = sobel_y.expand(3, 1, 3, 3).to(rgb)

        # --- 2. Apply the 2D convolutions -----------------------------------------
        # Kernel size is 3 ⇒ one‑pixel border is enough to keep size unchanged.
        grad_x = F.conv2d(rgb, weight_x, padding=1, groups=3)
        grad_y = F.conv2d(rgb, weight_y, padding=1, groups=3)

        # --- 3. Edge magnitude per channel ----------------------------------------
        # A small epsilon avoids a zero‑gradient sqrt warning.
        edges = torch.sqrt(grad_x**2 + grad_y**2 + 1e-6)
        # edges = grad_x + grad_y
        return edges


sobel = Sobel().to('mps').to(torch.float32)
sm = torch.jit.script(sobel)
sm.save("models/sobel_edge_float32.pt")

sm = torch.jit.load("models/sobel_edge_float32.pt")
# sm = torch.jit.load("models/mosaic_float32.pt")
# sm.to('mps')

mosaic = torch.jit.load("models/mosaic_float16.pt")
mosaic.to('mps')

# print(sm)

import sys
# sys.exit(0)
import time

ticks = 1

while True:
    ret, frame_bgr = cam.read()

    # Write the frame to the output file
    # out.write(frame)

    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    # 3) Ensure the array is contiguous (torch needs it) -------------------------
    frame_rgb = np.ascontiguousarray(frame_rgb)

    # 4) numpy -> torch, move channels, scale, add batch if wanted --------------
    tensor = torch.from_numpy(frame_rgb)     # H x W x C, uint8 → int tensor
    tensor = tensor.to("mps", non_blocking=True)
    
    tensor = tensor.permute(2, 0, 1)         # C x H x W
    tensor = tensor.to(torch.float32).div(255.0)       # float32, [0,1]

    # normalize tensor to ImageNet mean and std
    # mean = (0.485, 0.456, 0.406)  # ImageNet defaults (RGB)
    # std  = (0.229, 0.224, 0.225)
    # mean = torch.tensor(mean, dtype=tensor.dtype, device=tensor.device)[:, None, None]
    # std  = torch.tensor(std,  dtype=tensor.dtype, device=tensor.device)[:, None, None]
    # tensor.sub_(mean).div_(std)    

    # 5) (Optional) add a batch dim and push to GPU ------------------------------
    tensor = tensor.unsqueeze(0)             # 1 x C x H x W

    # if ticks == 3:
    #     tensor = tensor.to(torch.float16)
    #     mosaic = torch.jit.load("models/mosaic_float16.pt")
    #     mosaic.to('mps')
    #     mosaic_output = mosaic(tensor) / 255.0
    #     # mosaic_output = undo_normalize(mosaic_output)
    #     print('input:',tensor.shape,tensor.dtype)
    #     print('mosaic output:',mosaic_output.shape,mosaic_output.dtype)
    #     torchvision.utils.save_image(tensor[0], 'input_tensor.png')
    #     torchvision.utils.save_image(mosaic_output[0], 'mosaic_output.png')

    #     sys.exit(0)

    output_tensor = sm(tensor.to(torch.float16))
    # print('input:',tensor.shape,tensor.dtype)
    # print('output:',output_tensor.shape)


    # frame_bgr_out = tensor_to_bgr(output_tensor, undo_normalise=True,mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    frame_bgr_out = tensor_to_bgr(output_tensor)

    # Display the captured frame
    cv2.imshow('Camera', frame_bgr_out)

    # time.sleep(1.0)

    # Press 'q' to exit the loop
    # if ticks > 10:
    #     break

    if cv2.waitKey(1) == ord('q'):
        break
    ticks += 1

# Release the capture and writer objects
cam.release()
# out.release()
cv2.destroyAllWindows()