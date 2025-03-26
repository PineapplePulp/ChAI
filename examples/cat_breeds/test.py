import torch
import utils
import os

datapath = '/Users/iainmoncrief/Documents/Github/ChAI/examples/cat_breeds/data/catbreeds'
loader = utils.cat_breed_dataset(datapath)

print(loader[10])