import torch
import os
# from torch.utils.data import Dataset
from torchvision.datasets import VisionDataset
import numpy as np
import matplotlib.pyplot as plt

# for https://www.kaggle.com/datasets/imbikramsaha/cat-breeds/data
class cat_breed_dataset(VisionDataset):
    def __init__(self, path_to_data):
        self.imgpath = os.path.join(path_to_data, "images")
        self.labpath = os.path.join(path_to_data, "labels")
        self.images, self.labels = [], []
        for lab in os.listdir(self.labpath):
            if "item" in lab:
                self.labels.append(
                    np.load(os.path.join(self.labpath, lab))
                )
        self.labels = np.array(self.labels)
        for img in os.listdir(self.imgpath):
            if "item" in img:
                self.images.append(
                    np.load(os.path.join(self.imgpath, img))
                )
        self.images = np.array(self.images)

        assert len(self.images) == len(self.labels)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img = torch.tensor(self.images[idx]).float()
        lab = torch.tensor(self.labels[idx]).long()
        # `tensor` is lowercase to make `lab` a 0-dim tensor
        return img, lab
    
def train(model, device, train_loader, optimizer, criterion, epoch, lambda_reg=0.01, one_pass=False, verbose=False):
    model.train()
    avg_loss = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)

        loss = criterion(output, target)
        avg_loss += loss
        loss.backward()

        optimizer.step()
        if one_pass: break
    
    avg_loss /= len(train_loader.dataset)

    if verbose:
        print(f'Train Epoch: {epoch} \tAverage loss: {avg_loss:.6f}')