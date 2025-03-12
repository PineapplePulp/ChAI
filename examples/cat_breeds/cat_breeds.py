import os
from pathlib import Path

import torch
from torchvision.datasets import VisionDataset
from torch.utils.data import Dataset

# import matplotlib.pyplot as plt

import numpy as np
import json

my_dir = Path(__file__).resolve().parent

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
    def __init__(self, data_dir, download=False):
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
    
    # def download(self):


if __name__ == "__main__":
    my_dir = Path(__file__).resolve().parent
    data_dir = my_dir / 'data'
    cat_breeds = CatBreedsData(data_dir)
    