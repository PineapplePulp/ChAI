import kagglehub

# Download latest version
path = kagglehub.dataset_download("imbikramsaha/cat-breeds")

print("Path to dataset files:", path)