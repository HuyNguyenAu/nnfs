# %%
import os
import requests
from zipfile import ZipFile
from cv2 import imread, IMREAD_UNCHANGED
from glob2 import iglob
import numpy as np

url: str = 'https://nnfs.io/datasets/fashion_mnist_images.zip'
file: str = 'fashion_mnist_images.zip'
folder: str = 'fashion_mnist_images'

# Download the MNIST images if not found.
if not os.path.isfile(file):
    print(f'Downloading {url}...')

    response: requests.Response = requests.get(url)

    # Write the file to disk.
    with open(file, 'wb') as file_stream:
        file_stream.write(response.content)

# Extract the file to the fashion_mnist_images dir.
if not os.path.isdir(folder):
    print(f'Extracting {file} to {folder}...')
    
    with ZipFile(file, 'r') as zip_file:
        zip_file.extractall(folder)

x_train = []
y_train = []
x_val = []
y_val = []

# Find all files recursively and build up out training and validation data.
print(f'Searching for all images in {folder}...')
for file_path in iglob(f'{folder}/**/*.png', recursive=True):
    print(f'Loading {file_path}...')
    items: list[str] = file_path.split('/')
    image: np.ndarray = imread(file_path, IMREAD_UNCHANGED)
    label: int = int(items[-2])
    
    if 'train' in file_path:
        x_train.append(image)
        y_train.append(label)
    else:
        x_val.append(image)
        y_val.append(label)
# %%
