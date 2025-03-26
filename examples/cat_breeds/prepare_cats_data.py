import kagglehub

from pathlib import Path
import cv2
import numpy as np
import json
import shutil


my_dir = Path(__file__).resolve().parent

output_dir = my_dir / 'data' / 'catbreeds'
output_dir.mkdir(parents=True, exist_ok=True)

img_dir = output_dir / 'images'
img_dir.mkdir(parents=True, exist_ok=True)

label_dir = output_dir / 'labels'
label_dir.mkdir(parents=True, exist_ok=True)

raw_dir = output_dir / 'raw'
raw_dir.mkdir(parents=True, exist_ok=True)

label_names_path = output_dir / 'label_names.json'

# Download latest version
path = kagglehub.dataset_download('imbikramsaha/cat-breeds')

cat_breeds_dir = Path(path)
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
