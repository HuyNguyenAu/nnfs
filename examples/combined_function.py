# %%
from nnfs.datasets import spiral_data
import nnfs

from core import Models, Layers, ActivationFunctions, Optimisers, Accuracies, CombinedFunctions, Data

nnfs.init()

# Create data.
x, y = spiral_data(samples=5000, classes=3)

# Load the data into the data loader.
data_loader: Data.DataLoader = Data.DataLoader(x_train=x, y_train=y)

# Create a new model.
model = Models.Model()

# Add a dense layer with 2 input features and 128 output values.
model.add_layer(Layers.Dense(
    2, 128, weight_regulariser_l2=5e-4, bias_regulariser_l2=5e-4))

# Add a ReLU activation function.
model.add_layer(ActivationFunctions.ReLU())

# Add a dense layer with 128 input features and 128 output values.
model.add_layer(Layers.Dense(128, 128))

# Add a ReLU activation function.
model.add_layer(ActivationFunctions.ReLU())

# Add a dense layer with 128 input features and 3 output values.
model.add_layer(Layers.Dense(128, 3))

# Add the Softmax Categorical Cross Entropy combined loss and activation function.
model.add_layer(CombinedFunctions.SoftmaxCategoricalCrossEntropy())

# Set the optimiser to ADAM.
model.set_optimiser(Optimisers.Adam(learning_rate=0.0002, decay=5e-7))

# Set the accuracy as Categorical.
model.set_accuracy(Accuracies.Categorical())

# Finalise the model.
model.finalise()

# Perform model training.
model.train(data_loader.get_x_train(), data_loader.get_y_train(), epochs=500, print_every=100)

# Perfom model predication.
model.validate(data_loader.get_x_val(), data_loader.get_y_val())

# %%
