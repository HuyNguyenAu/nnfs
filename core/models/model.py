import numpy as np

from ..loss_functions import LossFunction
from ..activation_functions import ActivationFunction
from ..combined_functions import SoftmaxCategoricalCrossEntropy
from ..optimisers import Optimiser
from ..layers import Layer, Input, Dense, Dropout
from ..accuracy import Accuracy
from ..types import CombinedFunctionType


class Model:
    '''
    The model class.
    '''

    def __init__(self) -> None:
        self.layers: list[Layer] = []
        self.loss_function: LossFunction = None
        self.optimiser: Optimiser = None
        self.input_layer: Layer = None
        self.accuracy: Accuracy = None
        self.prediction_layer: ActivationFunction = None
        self.trainable_layers: list[int] = []
        self.dropout_layers: list[int] = []
        self.combined_function_type: CombinedFunctionType = None

    def add_layer(self, layer: Layer) -> None:
        '''
        Add a new layer to the model.
        '''
        self.layers.append(layer)

    def set_loss_function(self, loss_function: LossFunction) -> None:
        '''
        Set the loss function for the model.
        '''
        self.loss_function = loss_function

    def set_optimiser(self, optimiser: Optimiser) -> None:
        '''
        Set the optimiser for the model.
        '''
        self.optimiser = optimiser

    def set_accuracy(self, accuracy: Accuracy) -> None:
        '''
        Set the accuracy for the model.
        '''
        self.accuracy = accuracy

    def forward(self, x: np.ndarray) -> np.ndarray:
        '''
        Perform a forward pass.
        '''
        self.input_layer.forward(x)

        layer_count: int = len(self.layers)

        for i in range(layer_count):
            current_layer: Layer = self.layers[i]

            if i <= 0:
                current_layer.forward(self.input_layer.get_output())
            else:
                current_layer.forward(self.layers[i - 1].get_output())

        return self.layers[layer_count - 1].get_output()

    def backward(self, output: np.ndarray, y: np.ndarray) -> None:
        '''
        Perform a backward pass.
        '''
        self.loss_function.backward(output, y)

        layer_count: int = len(self.layers)

        for i in reversed(range(layer_count)):
            if i >= layer_count - 1:
                self.layers[layer_count -
                            1].backward(self.loss_function.get_d_inputs())
            else:
                self.layers[i].backward(self.layers[i + 1].get_d_inputs())

    def get_combined_function_type(self, combined_function: Layer) -> CombinedFunctionType:
        '''
        Determine the combined function type from the given combined function.
        '''
        if isinstance(combined_function, SoftmaxCategoricalCrossEntropy):
            return CombinedFunctionType.ACTIVATION_LOSS

        return None

    def finalise(self) -> None:
        '''
        Finalise the model.
        '''
        self.input_layer = Input()

        for i, _ in enumerate(self.layers):
            current_layer: Layer = self.layers[i]

            # Keep track of all trainable and dropout layers.
            # NOTE: For now, we will only be dealing with Dense layers. This should be fine for now.
            if current_layer.get_is_trainable() and isinstance(current_layer, Dense):
                self.trainable_layers.append(i)
            elif isinstance(current_layer, Dropout):
                self.dropout_layers.append(i)

            # Set the prediction layer.
            # If we come across a combined function, set the combined function type.
            if i >= len(self.layers) - 1:
                if isinstance(current_layer, SoftmaxCategoricalCrossEntropy):
                    self.loss_function = current_layer
                    self.prediction_layer = current_layer
                    self.combined_function_type = CombinedFunctionType.ACTIVATION_LOSS
                elif isinstance(current_layer, ActivationFunction):
                    self.prediction_layer = current_layer
                    print(isinstance(current_layer,
                          SoftmaxCategoricalCrossEntropy), current_layer)

    def calculate_regularisation_loss(self) -> float:
        '''
        Get the regularisation loss for the current epoch.
        '''
        regularisation_loss: float = 0
        for i in self.trainable_layers:
            regularisation_loss += self.loss_function.regularisation_loss(
                self.layers[i])

        return regularisation_loss

    def optimise(self) -> None:
        '''
        Perform optimisation.
        '''
        self.optimiser.pre_update_parameters()

        for i in self.trainable_layers:
            self.optimiser.update_parameters(self.layers[i])

        self.optimiser.post_update_parameters()

    def train(self, x_train: np.ndarray, y_train: np.ndarray, *, epochs: int = 1, print_every: int = 100) -> None:
        '''
        Train the model.
        '''
        self.accuracy.init(y=y_train)

        for epoch in range(0, epochs):
            # Perform a forwards pass.
            output = self.forward(x_train)

            data_loss: float = None

            # Calculate the losses.
            data_loss: float = self.loss_function.calculate(output, y_train)
            regularisation_loss: float = self.calculate_regularisation_loss()
            loss: float = regularisation_loss + data_loss

            # Get predictions.
            predictions = self.prediction_layer.prediction(output)

            # Calculate accuracy.
            accuracy: float = self.accuracy.calculate(
                predictions, y_train, self.combined_function_type)

            if not epoch % print_every:
                print(
                    f'Epoch: {epoch}, ' +
                    f'Accuracy: {accuracy:.3f}, ' +
                    f'Loss: {loss:.9f}, ' +
                    f'Data Loss: {data_loss:.9f}, ' +
                    f'Regularisation Loss: {regularisation_loss:.9f}, ' +
                    f'Learning Rate: {self.optimiser.get_current_learning_rate():.9f}'
                )

            # Perform a backwards pass.
            self.backward(output, y_train)

            # Perfom optimisations.
            self.optimise()

    def prediction(self, x: np.ndarray) -> np.ndarray:
        '''
        Perform model prediction.
        '''
        # Disable all dropout layers.
        for i in self.dropout_layers:
            self.layers[i].set_is_disabled(True)

        # Perform model prediction.
        output = self.forward(x)

        # Enable all dropout layers.
        for i in self.dropout_layers:
            self.layers[i].set_is_disabled(False)

        return output

    def validate(self, x_val: np.ndarray, y_val: np.ndarray) -> None:
        '''
        Perform model validation.
        '''
        output: np.ndarray = self.prediction(x_val)

        # Calculate model loss.
        loss = self.loss_function.calculate(output, y_val)

        # Get predictions.
        predictions = self.prediction_layer.prediction(output)

        # Calculate accuracy.
        accuracy: float = self.accuracy.calculate(
                predictions, y_val, self.combined_function_type)

        print(
            f'Validation Accuracy: {accuracy:.9f}, Validation Loss: {loss:.9f}')
