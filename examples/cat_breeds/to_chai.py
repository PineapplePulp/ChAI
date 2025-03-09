from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parents[1]))

import lib.chai
import torch
import os
import numpy as np

model = torch.load("./cat_breeds/models/pretest.pt")
model.chai_dump("./cat_breeds/models/chai_model", "SmallCNN")

# load_path = "./cat_breeds/data/catbreeds/images"
# for i, item in enumerate(os.listdir(load_path)):
#     if "item" in item: # check file name
#         img = np.load(f"{load_path}/{item}")
#         img = torch.Tensor(img)
#         img.chai_save("./cat_breeds/data/catbreeds/chai_images", f"item{i}", verbose=False)
#     if i > 20:
#         break