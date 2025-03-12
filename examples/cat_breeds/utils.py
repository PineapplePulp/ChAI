import torch
import os
# from torch.utils.data import Dataset
from torchvision.datasets import VisionDataset
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
import json


# for https://www.kaggle.com/datasets/imbikramsaha/cat-breeds/data
class cat_breed_dataset(VisionDataset):
    def __init__(self, path_to_data):
        data_dir = Path(path_to_data).resolve()

        img_dir = data_dir / 'images'
        label_dir = data_dir / 'labels'
        label_names_path = data_dir / 'label_names.json'

        labels = []
        for label_path in label_dir.iterdir():
            if 'item' in label_path.name:
                label_data = np.load(label_path)
                labels.append(label_data)
        
        images = []
        for img_path in img_dir.iterdir():
            if 'item' in img_path.name:
                img_data = np.load(img_path)
                images.append(img_data)

        label_names = {}
        with open(label_names_path, 'r') as f:
            label_names = json.load(f)
        
        self.labels = np.array(labels)
        self.images = np.array(images)
        self.label_names = label_names

        assert len(self.images) == len(self.labels) and len(set(self.labels)) == len(self.label_names)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        image = torch.tensor(self.images[idx]).float()
        label = torch.tensor(self.labels[idx]).long()
        # `tensor` is lowercase to make `lab` a 0-dim tensor
        return (image,label)
    
    def get_class_name(self,label):
        return self.label_names[label]
    
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