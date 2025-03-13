import torch
from torchvision.io import read_image, ImageReadMode
import sys
import os
from torchvision.transforms import Resize, Normalize

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'scripts')))
import chai

from pathlib import Path


def process(img_path):
    img = read_image(img_path,mode=ImageReadMode.RGB).float()
    print(img.shape, img.dtype)
    # write the image tensor to a file

    img = Resize((224, 224))(img) # resize
    img = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img / 255.0) # normalize
    print(img.shape, img.dtype)

    os.makedirs('imgs', exist_ok=True)
    # remove the extension
    img.chai_save('imgs', os.path.splitext(os.path.basename(img_path))[0])

dirs = []

for img_path in sys.argv[1:]:
    pth = Path(img_path).parent
    dirs.append(pth)

image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']

for dir in dirs:
    for img_path in dir.iterdir():
        if img_path.suffix in image_extensions:
            process(img_path)
