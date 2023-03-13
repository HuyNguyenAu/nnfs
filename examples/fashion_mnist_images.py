# %%
import os
import requests
from zipfile import ZipFile
from cv2 import imread, IMREAD_UNCHANGED
from multiprocessing import Pool
from glob2 import iglob
import numpy as np
from core import Models, Layers, ActivationFunctions, CombinedFunctions, Optimisers, Accuracies, Data

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

# Create the training and validation data arrays.
x_train = []
y_train: list[int] = []
x_val = []
y_val: list[int] = []

def task(path: str) -> tuple[str, int, np.ndarray]:
    '''
    '''
    items: list[str] = path.split('/')
    return (path, int(items[-2]), imread(path, IMREAD_UNCHANGED))

# Load all images in parallel.
with Pool(10) as pool:
    # Find all files recursively and build up out training and validation data.
    print(f'Searching for all images in {folder}...')
    image_paths: list[str] = iglob(f'{folder}/**/*.png', recursive=True)

    for file_path, label, image in pool.map_async(task, image_paths).get():
        print(f'Loading {file_path}...')

        if 'train' in file_path:
            x_train.append(image)
            y_train.append(label)
        else:
            x_val.append(image)
            y_val.append(label)

    # Convert all training and validation images to 32-bit floats.
    x_train = np.array(x_train, dtype=np.float32)
    y_train = np.array(y_train).astype('uint8')

    x_val = np.array(x_val, dtype=np.float32)
    y_val = np.array(y_val).astype('uint8')

# Flatten the features.
x_train = x_train.reshape(x_train.shape[0], -1)
x_val = x_val.reshape(x_val.shape[0], -1)

# Scale features to between -1.0 and 1.0.
x_train = (x_train - 127.5) / 127.5
x_val = (x_val - 127.5) / 127.5


# Load the data into the data loader.
data_loader: Data.DataLoader = Data.DataLoader(
    x_train=x_train, y_train=y_train, x_val=x_val, y_val=y_val)

# Shuffle the training dataset.
data_loader.shuffle()

model = Models.Model()
model.add_layer(Layers.Dense(x_train.shape[1], 128))
model.add_layer(ActivationFunctions.ReLU())
model.add_layer(Layers.Dense(128, 128))
model.add_layer(ActivationFunctions.ReLU())
model.add_layer(Layers.Dense(128, 10))
model.add_layer(CombinedFunctions.SoftmaxCategoricalCrossEntropy())
model.set_optimiser(Optimisers.Adam(decay=1e-3))
model.set_accuracy(Accuracies.Categorical())

model.finalise()
model.train(x_train=x_train, y_train=y_train,
            epochs=10, batch_size=128, print_every=100)
