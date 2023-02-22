from nnfs.datasets import spiral_data
import nnfs

from core import Models, Layers, ActivationFunctions, LossFunctions, Optimisers, Accuracies, CombinedFunctions

nnfs.init()

# Create training dataset.
x_train, y_train = spiral_data(samples=1000, classes=3)

# Create a new model.
model = Models.Model()

# Add a dense layer with 1 input features and 64 output values.
model.add_layer(Layers.Dense(
    2, 64, weight_regulariser_l2=5e-4, bias_regulariser_l2=5e-4))

# Add a ReLU activation function.
model.add_layer(ActivationFunctions.ReLU())

# Add a drop out layer.
model.add_layer(Layers.Dropout(rate=0.1))

# Add a dense layer with 64 input features and 3 output values.
model.add_layer(Layers.Dense(64, 3))

# Add a Sigmoid activation function.
model.add_layer(ActivationFunctions.Softmax())

model.set_loss_function(LossFunctions.CategoricalCrossEntropy())

# Set the optimiser to ADAM.
model.set_optimiser(Optimisers.Adam(learning_rate=0.05, decay=5e-5))

# Set the accuracy.
model.set_accuracy(Accuracies.Categorical())

# Finalise the model.
model.finalise()

# Perform model training.
model.train(x_train, y_train, epochs=10000, print_every=100)

# # Create the test dataset.
# x_test, y_test = spiral_data(samples=1000, classes=3)

# # Perfom model predication.
# model.validate(x_val=x_test, y_val=y_test)
