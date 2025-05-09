import argparse
import os
import sys
import time
from pathlib import Path


import numpy as np
import torch
import torch.nn as nn
import torchvision
import cv2
import utils

def dir_path(path):
    if os.path.isdir(path):
        return path
    else:
        raise argparse.ArgumentTypeError(f'readable_dir:{path} is not a valid path')

def torch_device(device_name):
    if device_name == 'cpu':
        return torch.device('cpu')
    elif device_name == 'cuda':
        if torch.cuda.is_available():
            return torch.device('cuda')
        else:
            raise argparse.ArgumentTypeError(f'cuda is not available')
    elif device_name == 'mps':
        if torch.backends.mps.is_available():
            return torch.device('mps')
        else:
            raise argparse.ArgumentTypeError(f'mps is not available')
    elif device_name == None:
        if torch.backends.mps.is_available():
            return torch.device('mps')
        elif torch.cuda.is_available():
            return torch.device('cuda')
        else:
            return torch.device('cpu')
    else:
        raise argparse.ArgumentTypeError(f'unknown device name: {device_name}')

parser = argparse.ArgumentParser(description='Process files in a directory.')
parser.add_argument('--device', dest='device', type=torch_device, default=None,
                    help='Device to use for computation (default: cpu).')

parser.add_argument('--model-file', type=Path, required=True,
                    help='Path to the model file (e.g., .pt).')

parser.add_argument('--use-webcam', action='store_true',
                    help='Use webcam for input (default: False).')

parser.add_argument('--input-video-file', type=Path, help='Path to the input video file (default: webcam).', default=None)
parser.add_argument('--output-video-file', type=Path, help='Path to the output video file (default: webcam).', default=None)

parser.add_argument('--show-output', action='store_true',
                    help='Show output video in a window (default: False).')


args = parser.parse_args()

arg_dict = vars(args)
for arg in arg_dict:
    print(f'args.{arg}: {arg_dict[arg]}')

if args.use_webcam or args.input_video_file:
    if args.input_video_file and args.use_webcam:
        raise argparse.ArgumentTypeError('Cannot use both webcam and input video file at the same time.')
    if args.input_video_file:
        print('using input video file:', args.input_video_file)
        args.use_webcam = False
    else:
        args.input_video_file = None
        args.use_webcam = True
        print('using webcam for input video')



default_device = args.device
if default_device is None:
    default_device = torch.device('cpu')
    if torch.backends.mps.is_available():
        default_device = torch.device('mps')
        print('using mps')

    if torch.cuda.is_available():
        default_device = torch.device('cuda')
        print('using cuda')

print('using default device:', default_device)

torch.set_default_device(default_device)


# Open the default camera
cam = cv2.VideoCapture(str(args.input_video_file) if args.input_video_file else 0)
if not cam.isOpened():
    print("Error: Could not open video.")
    sys.exit()

# Get the default frame width and height
frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
capture_fps = int(cam.get(cv2.CAP_PROP_FPS))
# Define the codec and create VideoWriter object
if args.output_video_file:
    if not args.output_video_file.exists():
        args.output_video_file.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = out = cv2.VideoWriter(str(args.output_video_file), fourcc, capture_fps, (frame_width, frame_height))



model = torch.jit.load(str(args.model_file))
model = model.to(default_device)
model.eval()
print('Loaded model:', args.model_file)


done_writing_to_output = False

while True:
    ret, frame_bgr = cam.read()
    if not ret:
        if args.use_webcam:
            print("Error: Could not read frame from webcam.")
        if args.input_video_file and args.show_output:
            done_writing_to_output = True
            cam = cv2.VideoCapture(str(args.input_video_file))
            ret, frame_bgr = cam.read()
            if not ret:
                print("Error: Could not read frame from input video file.")
                break
            else:
                continue
        else:
            break



    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    # 3) Ensure the array is contiguous (torch needs it) -------------------------
    frame_rgb = np.ascontiguousarray(frame_rgb)

    # 4) numpy -> torch, move channels, scale, add batch if wanted --------------
    tensor = torch.from_numpy(frame_rgb)     # H x W x C, uint8 → int tensor
    tensor = tensor.to(default_device, non_blocking=True)
    
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


    if args.model_file.name == 'sobel_edge_float32.pt':
        output_tensor = model(tensor.to(torch.float16))
    else:
        output_tensor = model(tensor.to(torch.float16)) / 255.0
    # print('input:',tensor.shape,tensor.dtype)
    # print('output:',output_tensor.shape)


    # frame_bgr_out = tensor_to_bgr(output_tensor, undo_normalise=True,mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    frame_bgr_out = utils.tensor_to_bgr(output_tensor)


    if args.show_output or args.use_webcam:
        cv2.imshow('Frame', frame_bgr_out)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    if args.output_video_file and done_writing_to_output:
        out.write(frame_bgr_out)


# Release the capture and writer objects
if args.output_video_file:
    out.release()
if args.use_webcam:
    cam.release()
if args.show_output:
    cv2.destroyAllWindows()
