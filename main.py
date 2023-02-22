# %%

import numpy as np
import matplotlib.pyplot as plt
import nnfs
from nnfs.datasets import sine_data

from core import Models, Layers, ActivationFunctions, LossFunctions, Optimisers, Accuracies

nnfs.init()

# Create training dataset.
x_train, y_train = sine_data()

# Create a new model.
model = Models.Model()

# Add a dense layer with 2 input features and 64 output values.
model.add_layer(Layers.Dense(1, 64))

# Add a ReLU activation function.
model.add_layer(ActivationFunctions.ReLU())

# Add a dense layer with 64 input features and 64 output values.
model.add_layer(Layers.Dense(64, 64))

# Add a ReLU activation function.
model.add_layer(ActivationFunctions.ReLU())

# Add a dense layer with 64 input features and 1 output values.
model.add_layer(Layers.Dense(64, 1))

# # Add a linear activation function.
model.add_layer(ActivationFunctions.Linear())

# Set the loss function to mean squared error.
model.set_loss_function(LossFunctions.MeanSquaredError())

# Set the optimiser to ADAM.
model.set_optimiser(Optimisers.Adam(learning_rate=0.005, decay=1e-3))

# Set the accuracy.
model.set_accuracy(Accuracies.Regression())

# Finalise the model.
model.finalise()

# Perform model training.
model.train(x_train, y_train, epochs=500, print_every=100)

# Create the test dataset.
x_test, y_test = sine_data()

# Perfom model validation.
output_test = model.predict(x_test)

plt.plot(x_test, y_test)
plt.plot(x_test, output_test)
plt.show()