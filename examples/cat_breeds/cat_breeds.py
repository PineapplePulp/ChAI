import os
from pathlib import Path

import torch
import torch.nn as nn
from torchvision.datasets import VisionDataset
from torch.utils.data import Dataset, DataLoader

# import matplotlib.pyplot as plt

import numpy as np
import json

my_dir = Path(__file__).resolve().parent

def get_best_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')

def download_cat_breeds_data(data_dir):
    import kagglehub
    import cv2

    data_dir = Path(data_dir).resolve()
    data_dir.mkdir(parents=True, exist_ok=True)

    output_dir = data_dir / 'catbreeds'
    output_dir.mkdir(parents=True, exist_ok=True)

    img_dir = output_dir / 'images'
    img_dir.mkdir(parents=True, exist_ok=True)

    label_dir = output_dir / 'labels'
    label_dir.mkdir(parents=True, exist_ok=True)

    raw_dir = output_dir / 'raw'
    raw_dir.mkdir(parents=True, exist_ok=True)

    label_names_path = output_dir / 'label_names.json'

    # Download latest version
    dl_path = kagglehub.dataset_download('imbikramsaha/cat-breeds')

    cat_breeds_dir = Path(dl_path).resolve()
    print("Path to dataset files:", cat_breeds_dir)

    cat_classes_dir = cat_breeds_dir / 'cats-breads'

    classes = sorted([c for c in cat_classes_dir.iterdir() if c.is_dir()])

    label_names = {}

    n = 0
    for i, c in enumerate(classes):
        print(f'Processing label: {i}, class: {c.name}')
        label_names[i] = c.name

        raw_class_dir = raw_dir / 'resized' / c.name
        raw_class_dir.mkdir(parents=True, exist_ok=True)

        files = [f for f in c.iterdir() if f.is_file()]
        for f in files:
            image = cv2.imread(str(f))
            if image is None:
                # Skip corrupted or non-image fpaths
                continue
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            resized_img = cv2.resize(
                src=image,
                dsize=(32, 32),
                interpolation=cv2.INTER_CUBIC
            )
            transposed_img = resized_img.transpose(2, 0, 1)

            img_path = img_dir / f'item{n}'
            label_path = label_dir / f'item{n}'
            raw_img_path = raw_class_dir / str(f.name)

            np.save(str(img_path.resolve()), transposed_img)
            np.save(str(label_path.resolve()), i)
            cv2.imwrite(str(raw_img_path.resolve()), resized_img) # Only save resized image

            n += 1

    with open(label_names_path, 'w') as f:
        json.dump(label_names, f)

    download_dir = cat_breeds_dir.parent.parent.parent.parent.resolve()
    # print(f'Deleting downloaded dataset files in {download_dir}')
    # shutil.rmtree(download_dir)

    return output_dir



# for https://www.kaggle.com/datasets/imbikramsaha/cat-breeds/data
class CatBreedsData(VisionDataset):
    def __init__(self, data_dir, download=False, device=None):
        if download:
            download_cat_breeds_data(data_dir)
        data_dir = Path(data_dir / 'catbreeds').resolve()

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
        
        # if device is None:
        #     device = torch.get_default_device()
        # self.device = device

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
    
    def get_label_name(self,idx):
        if idx in self.label_names:
            return self.label_names[idx].lower()
        raise ValueError(f'Label index {idx} not found in label names.')
    
    def get_label_idx(self,label):
        for k,v in self.label_names.items():
            if v.lower() == label.lower():
                return k
        raise ValueError(f'Label {label} not found in label names.')


class SmallCNN(nn.Module):
    def __init__(self):
        super(SmallCNN, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 128, 3, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(128, 256, 3, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.LazyLinear(2048),
            nn.ReLU(),
            nn.LazyLinear(512),
            nn.ReLU(),
            nn.LazyLinear(12),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        return self.layers(x)

def train(model, 
          device, 
          train_loader, 
          optimizer, 
          criterion, 
          epoch, 
          lambda_reg=0.01,
          one_pass=False,
          verbose=False):
    
    # print('Training model...')
    model.train()
    avg_loss = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.to(device)
        target = target.to(device)
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
    return avg_loss

def eval(model, 
          device, 
          test_loader, 
          optimizer, 
          criterion, 
          epoch):
    
    # print('Evaluating model...')
    model.eval()
    avg_loss = 0

    for batch_idx, (data, target) in enumerate(test_loader):
        data = data.to(device)
        target = target.to(device)
        output = model(data)
        loss = criterion(output, target)
        avg_loss += loss
    
    avg_loss /= len(test_loader.dataset)

    # print(f'Test Epoch: {epoch} \tAverage loss: {avg_loss:.6f}')
    return avg_loss


# def eval(model, 
#          device, 
#          test_loader):
    
#     print('Evaluating model...')
#     model.eval()
#     test_loss = 0
#     correct = 0

#     with torch.no_grad():
#         for data, target in test_loader:
#             data = data.to(device)
#             target = target.to(device)
#             output = model(data)
#             test_loss += criterion(output, target).item()
#             pred = output.argmax(dim=1, keepdim=True)
#             correct += pred.eq(target.view_as(pred)).sum().item()

#     test_loss /= len(test_loader.dataset)

#     print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({100. * correct / len(test_loader.dataset):.0f}%)\n')


def get_attributes(model):
    return { str(k): str(v[0]) if isinstance(v,tuple) else str(v).lower() for (k,v) in dict(model.__dict__).items() if k[0] != '_'}

def get_summary(model,global_name,parent_name=None):
    model_name = model.__class__.__name__
    d = {
        'layerType': model_name,
        'attributes': get_attributes(model),
        'subModules': { name : get_summary(m,global_name=global_name,parent_name=name) for name,m in  model.named_children() },
        'subModuleOrder': [name for name,_ in model.named_children()]
    }
    return d

def has_same_architecture(model1, model2):
    return get_summary(model1,'') == get_summary(model2,'')

if __name__ == "__main__":
    my_dir = Path(__file__).resolve().parent
    data_dir = my_dir / 'data'
    model_path = my_dir / 'models' / 'pretest.pt'

    print('Constructing CatBreedsData Dataset.')
    data_set = CatBreedsData(data_dir)
    
    data_set_size = len(data_set)
    train_len = int(data_set_size*0.9)      
    test_len = data_set_size - train_len
    train_set, test_set = torch.utils.data.random_split(data_set, [train_len, test_len])

    print('Train set size:', len(train_set))
    print('Test set size:', len(test_set))

    print('Creating DataLoader(s).')
    train_batch_size = test_len
    train_loader = DataLoader(train_set, batch_size=train_batch_size, shuffle=True)

    test_batch_size = test_len
    test_loader = DataLoader(test_set, batch_size=test_batch_size, shuffle=True)

    test_features, test_labels = next(iter(test_loader))
    print(test_features[0].shape,test_labels[0])
    print(test_features[1].shape,test_labels[1])

    device = get_best_device()
    print('Using device:', device)

    print('Creating model...')
    model = SmallCNN()

    # if model_path.exists():
    #     print('Loading model from', model_path)
    #     model1 = torch.load(model_path)
    #     if has_same_architecture(model, model1):
    #         print('Model architecture match. Using existing model.')
    #         model = model1
    #     else:
    #         print('Model architecture mismatch. Using new model.')

    print('Moving model to device...')
    model.to(device)

    epochs = 400

    print('Constructing optimizer and criterion...')
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = torch.nn.CrossEntropyLoss()

    print('Starting training...')
    for epoch in range(epochs):
        train_loss = train(model, device, train_loader, optimizer, criterion, epoch, one_pass=False, verbose=False)
        eval_loss = eval(model, device, test_loader, optimizer, criterion, epoch)
        print(f'Epoch {epoch} \tTrain loss: {train_loss:.6f} \tTest loss: {eval_loss:.6f}')
        
    model.to(torch.device('cpu'))
    torch.save(model, model_path)
    print("Model saved to", my_dir / 'models' / 'pretest.pt')


    