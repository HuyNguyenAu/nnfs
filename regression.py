from nnfs.datasets import spiral_data
import nnfs

from core import Models, Layers, ActivationFunctions, LossFunctions, Optimisers, Accuracies

nnfs.init()

# Create training dataset.
x, y = spiral_data(samples=1000, classes=2)

# Reshape labels to be list of lists. Inner list contains binary values per neuron output.
# From [0, 0, 0] to [[0], [0], [0]].
y = y.reshape(-1, 1)

x_train, y_train = x[:500], y[:500]
x_val, y_val = x[500:], y[500:]


# Create a new model.
model = Models.Model()

# Add a dense layer with 1 input features and 64 output values.
model.add_layer(Layers.Dense(
    2, 64, weight_regulariser_l2=5e-4, bias_regulariser_l2=5e-4))

# Add a ReLU activation function.
model.add_layer(ActivationFunctions.ReLU())

# Add a dense layer with 64 input features and 1 output values.
model.add_layer(Layers.Dense(64, 1))

# Add a Sigmoid activation function.
model.add_layer(ActivationFunctions.Sigmoid())

model.set_loss_function(LossFunctions.BinaryCrossEntropy())

# Set the optimiser to ADAM.
model.set_optimiser(Optimisers.Adam(decay=5e-7))

# Set the accuracy.
model.set_accuracy(Accuracies.Categorical(binary=True))

# Finalise the model.
model.finalise()

# Perform model training.
model.train(x_train, y_train, epochs=500, print_every=100)

# Perfom model predication.
model.validate(x_val, y_val)
