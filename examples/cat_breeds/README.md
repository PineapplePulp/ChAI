# Cat breeds tutorial

Download the images from https://www.kaggle.com/datasets/imbikramsaha/cat-breeds/code, or with the following code.

```
kagglehub.dataset_download("imbikramsaha/cat-breeds")
```

This tutorial demonstrates training and loading a model and its data into ChAI code to be used for multi-locale inference. 

## Image Preprocessing (load_cats.py)

The structure of the data will vary between different Kaggle datasets. For this specific one, the data consists of .jpg images of 12 different cat breeds, with each breed separated into its own directory. The following code iterates through every image in this structure.

```
classes=sorted(os.listdir(sdir) )
n = 0
for i, c in enumerate(classes):
    cpath=os.path.join(sdir, c)
    files=os.listdir(cpath)        
    for f in files:
        fpath=os.path.join(cpath,f)
```

PyTorch expects images to have (C, H, W) dimensions, which stands for channel, height, and width. It is also much easier to work with images that have the same height and width as each other. Within the for-loops above, every image in the dataset is resized down to (32, 32), transposed from (H, W, C) to (C, H, W), and saved as a .npy file. Labels are also saved as .npy files.

```
        image = cv2.imread(fpath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        resized_img = cv2.resize(
            src=image,
            dsize(32, 32),
            interpolation=cv2.INTER_CUBIC
        )
        transposed_img = np.transpose(resized_img, (2, 0, 1))
        np.save(f"{save_path}/images/item{n}", transposed_img)
        np.save(f"{save_path}/labels/item{n}", i)
```

## Building the Model (models/for_cats.py)

To create a customized model in PyTorch, we create a class that inherits from nn.Module and provides its own __init__() and forward() functions. Define the layers of the model in __init__() and specify how the data will pass through the model in forward().

Although it is not required, using nn.Sequential helps to ensure that every layer or activation function of the model is readable by ChAI.

```
import torch
import torch.nn as nn
import torch.nn.functional as F

class SmallCNN(nn.Module):
    def __init__(self):
        super(SmallCNN, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding="same"),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding="same"),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(8192, 256),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        return self.layers(x)
```

## Loading Data (utils.py and train_cnn.py)

PyTorch provides the DataLoader class to shuffle and split data into batches for training. To use it, we implement a custom Dataset class to read, store, and retrieve our images. The following implementation of `cat_breed_dataset` iterates through every image in our dataset and saves them in an array for images and another array for labels. Here, we assume that in the directory pointed to by `path_to_data`, there are two directories, one holding the images and the other holding the labels, both ordered the same as the other with each image and data called "image#.npy", "#" being a number.

```
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
```

Next, we implement `__len__` and `__getitem__`. For the latter, we return the image and the label as two separate tensors. We ensure that the label contains long(s) and the image contains floats for compatibility with the model's weights and the loss function.

```
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img = torch.tensor(self.images[idx]).float()
        lab = torch.tensor(self.labels[idx]).long()
        # `tensor` is lowercase to make `lab` a 0-dim tensor
        return img, lab
```

This class is instantiated and passed to a DataLoader for training.

```
cats_train = utils.cat_breed_dataset("./cat_breeds/data/catbreeds")
trainloader = DataLoader(cats_train, batch_size=128, shuffle=True)
```

## Training the Model (utils.py and train_cnn.py)

Before training, define a loss function and an optimizer to train the model.

```
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
criterion = torch.nn.CrossEntropyLoss()
```

During training, loss is calculated with the model's predictions and the provided labels. The model backpropagates the prediction error for the current batch, adjusting its parameters, before going to the next batch of data until the training is complete.

```
def train(model, device, train_loader, optimizer, criterion):
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
        print(f'Average loss: {avg_loss:.6f}')
```

Next, train the model and save it as a .pt file.

for epoch in range(epochs):
    utils.train(model, device, trainloader, optimizer, criterion, epoch, one_pass=False, verbose=True)

model.to(torch.device("cpu"))
torch.save(model, "./cat_breeds/models/pretest.pt")

## From PyTorch to ChAI (to_chai.py)

Once the images, labels, and the model have been saved as .npy and .pt files, we can call .chai_dump and .chai_save to save them as files that are readable by the current ChAI functionality. The following saves the first 20 images for brevity.

```
import lib.chai
import torch
import os
import numpy as np

model = torch.load("./cat_breeds/models/pretest.pt")
model.chai_dump("./cat_breeds/models/chai_model", "SmallCNN")

load_path = "./cat_breeds/data/catbreeds/images"
for i, item in enumerate(os.listdir(load_path)):
    if "item" in item: # check file name
        img = np.load(f"{load_path}/{item}")
        img = torch.Tensor(img)
        img.chai_save("./cat_breeds/data/catbreeds/chai_images", f"item{i}", verbose=False)
    if i > 20:
        break
```

The specific path that we follow here holds the data and the model in separate directories, as follows.

cat_breeds
├───models
│   ├───chai_model
│   │   ├───conv1.bias.chdata
│   │   ├───conv2.bias.json
│   │   └───...
│   └───pretest.pt
└───data
    └───catbreeds
        ├───chai_images
        │   ├───item0.chdata
        │   ├───item0.json
        │   └───...
        ├───images
        │   ├───item0.npy
        │   ├───item1.npy
        │   └───...
        └───labels
            ├───item0.npy
            ├───item1.npy
            └───...

## Single-locale inference in ChAI (single_locale.chpl)

We can call `loadModel` to read the model's information into ChAI.

```
var model: owned Module(real(32))  = loadModel(
    specFile="./cat_breeds/models/chai_model/specification.json",
    weightsFolder = "./cat_breeds/models/chai_model/",
    dtype=real(32)
);

writeln(model.signature);
```

Next, we can call Tensor.load to read each images' data into ChAI. The following code reads `numImages` images into an array.

```
config const numImages = 1;
var images = forall i in 0..<numImages do Tensor.load("./cat_breeds/data/catbreeds/chai_images/item"+i:string+".chdata") : real(32);
```

Lastly, we can use the model by passing images into it, which will call its forward function. The following code passes `numImages` images into the model `numTimes` times.

```
var preds: [0..<numImages] int;
config const numTimes = 1;
var time: real;
for i in 0..<numTimes {
    writeln("Inference (loop ",i,")...");
    var st = new Time.stopwatch();

    st.start();
    forall (img, pred) in zip(images, preds) {
        writeln(img.type:string);
        pred = model(img).argmax();
    }
    st.stop();

    const tm = st.elapsed();
    writeln("Time: ", tm, " seconds.");
}
```