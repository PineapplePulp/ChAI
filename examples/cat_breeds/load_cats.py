import os
import cv2
import numpy as np

# The next 8 lines were taken and modified from Kaggle
sdir=r'./example/cat-breeds/versions/1/cats-breads'
classes=sorted(os.listdir(sdir) )
n = 0
rejects = []
for i, c in enumerate(classes):
    cpath=os.path.join(sdir, c)
    files=os.listdir(cpath)        
    for f in files:
        fpath=os.path.join(cpath,f)
        
        # Use cv2 to resize images and use numpy to save them
        image = cv2.imread(fpath)
        if image is None:
            # Skip corrupted or non-image fpaths
            rejects.append(fpath)
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        resized_img = cv2.resize(
            src=image,
            dsize=(32, 32),
            interpolation=cv2.INTER_CUBIC
        )
        transposed_img = np.transpose(resized_img, (2, 0, 1))

        np.save(f'./example/data/catbreeds/images/item{n}', transposed_img)
        np.save(f'./example/data/catbreeds/labels/item{n}', i)
        n += 1

np.save(f'./example/data/catbreeds/', rejects)