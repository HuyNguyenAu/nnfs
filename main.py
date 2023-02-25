# %% Regression Example

import numpy as np
import matplotlib.pyplot as plt
import nnfs
from nnfs.datasets import sine_data

from core import Models, Layers, ActivationFunctions, LossFunctions, Optimisers, Accuracies

nnfs.init()

# Create training dataset.
x, y = sine_data()
x_train, y_train = x[:700], y[:700]
x_val, y_val = x[300:], y[300:]

# Create a new model.
model = Models.Model()

# Add a dense layer with 2 input features and 512 output values.
model.add_layer(Layers.Dense(1, 512))

# Add a ReLU activation function.
model.add_layer(ActivationFunctions.Sigmoid())

# Add a dense layer with 512 input features and 512 output values.
model.add_layer(Layers.Dense(512, 512))

# Add a ReLU activation function.
model.add_layer(ActivationFunctions.Sigmoid())

# Add a dense layer with 512 input features and 1 output values.
model.add_layer(Layers.Dense(512, 1))

# # Add a linear activation function.
model.add_layer(ActivationFunctions.Linear())

# Set the loss function to mean squared error.
model.set_loss_function(LossFunctions.MeanSquaredError())

# Set the optimiser to ADAM.
model.set_optimiser(Optimisers.Adam(learning_rate=0.005, decay=1e-3))

# Set the accuracy.
model.set_accuracy(Accuracies.Regression(y_train))

# Finalise the model.
model.finalise()

# Perform model training.
model.train(x_train, y_train, epochs=500, print_every=100)

# Perfom model validation.
output_test = model.prediction(x_val)

plt.plot(x_train, y_train)
plt.plot(x_val, output_test)
plt.show()


# %% Combined Function Example
from nnfs.datasets import spiral_data
import nnfs

from core import Models, Layers, ActivationFunctions, Optimisers, Accuracies, CombinedFunctions

nnfs.init()

# Create training and validation dataset.
x, y = spiral_data(samples=1000, classes=3)
x_train, y_train = x[:700], y[:700]
x_val, y_val = x[300:], y[300:]

# Create a new model.
model = Models.Model()

# Add a dense layer with 2 input features and 64 output values.
model.add_layer(Layers.Dense(
    2, 64, weight_regulariser_l2=5e-4, bias_regulariser_l2=5e-4))

# Add a ReLU activation function.
model.add_layer(ActivationFunctions.ReLU())

# Add a dense layer with 64 input features and 3 output values.
model.add_layer(Layers.Dense(64, 3))

# Add the Softmax Categorical Cross Entropy combined loss and activation function.
model.add_layer(CombinedFunctions.SoftmaxCategoricalCrossEntropy())

# Set the optimiser to ADAM.
model.set_optimiser(Optimisers.Adam(learning_rate=0.0002, decay=5e-7))

# Set the accuracy as Categorical.
model.set_accuracy(Accuracies.Categorical())

# Finalise the model.
model.finalise()

# Perform model training.
model.train(x_train, y_train, epochs=500, print_every=100)

# Perfom model predication.
model.validate(x_val, y_val)
