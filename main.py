# %%

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
# %%
