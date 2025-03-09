#%% Initialize loader and model
import torch
import utils
import os

modelpath = os.path.join(
    os.path.dirname(__file__),
    "models",
    "pretest.pt"
)

datapath = os.path.join(
    os.path.dirname(__file__),
    "data",
    "catbreeds"
)

model = torch.load(modelpath)
model.eval
loader = utils.cat_breed_dataset(datapath)
#%% Run an example
x, target = loader[103]
y = torch.argmax(model(x))
print(y)
print(target.item())
