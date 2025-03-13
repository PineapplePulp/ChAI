import torch.nn as nn
import torch
import sys
import os


from vgg import vgg16

model = vgg16(pretrained=True)

model = model.to(torch.float16)

# Add the scripts directory to the sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'scripts')))

import chai

os.makedirs('models/vgg16', exist_ok=True)
model.chai_dump('models/vgg16','vgg16', with_json=False, verbose=True)

# print(model.state_dict().keys())
# print([(n,w.dtype) for (n,w) in model.state_dict().items()])
