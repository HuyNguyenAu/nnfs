import numpy as np

import math

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

    def calculate_regularisation_loss(self) -> np.number:
        '''
        Get the regularisation loss for the current epoch.
        '''
        regularisation_loss: np.number = 0
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

    def train(self, x_train: np.ndarray, y_train: np.ndarray, *, epochs: int = 1, batch_size: int = None, print_every: int = 100) -> None:
        '''
        Train the model.
        '''
        # The default value if batch size is not set.
        steps: int = 1

        if batch_size is not None:
            steps = math.ceil(len(x_train) / batch_size)

        for epoch in range(0, epochs):
            self.loss_function.reset_accumulation()
            self.accuracy.reset_accumulation()

            for step in range(steps):
                batch_x_train: np.ndarray = x_train
                batch_y_train: np.ndarray = y_train

                if batch_size is not None:
                    batch_x_train = x_train[step *
                                            batch_size: (step + 1) * batch_size]
                    batch_y_train = y_train[step *
                                            batch_size: (step + 1) * batch_size]

                # Perform a forwards pass.
                output = self.forward(batch_x_train)

                data_loss: np.number = None

                # Calculate the losses.
                data_loss: np.number = self.loss_function.calculate(
                    output, batch_y_train)
                regularisation_loss: np.number = self.calculate_regularisation_loss()
                loss: np.number = regularisation_loss + data_loss

                # Get predictions.
                predictions = self.prediction_layer.prediction(output)

                # Calculate accuracy.
                accuracy: np.number = self.accuracy.calculate(
                    predictions, batch_y_train, self.combined_function_type)

                # Perform a backwards pass.
                self.backward(output, batch_y_train)

                # Perfom optimisations.
                self.optimise()

                if step <= steps:
                    print(
                        f'Epoch: {epoch}, ' +
                        f'Step: {step}, ' +
                        f'Accuracy: {accuracy:.3f}, ' +
                        f'Loss: {loss:.9f}, ' +
                        f'Data Loss: {data_loss:.9f}, ' +
                        f'Regularisation Loss: {regularisation_loss:.9f}, ' +
                        f'Learning Rate: {self.optimiser.get_current_learning_rate():.9f}'
                    )

            if batch_size is None:
                continue

            # Calculate the accumulated loss.
            accumulated_data_loss: np.number = self.loss_function.calculate_accumulated_loss()
            accumulated_regularisation_loss: np.number = self.calculate_regularisation_loss()
            accumulated_loss: np.number = accumulated_data_loss + \
                accumulated_regularisation_loss

            # Calculate the accumulated accuracy.
            accumulated_accuracy: np.number = self.accuracy.calculate_accumulated_accuracy()

            print(
                f'Accumulated Accuracy: {accumulated_accuracy:.3f}, ' +
                f'Accumulated Loss: {accumulated_loss:.9f}, ' +
                f'Accumulated Data Loss: {accumulated_data_loss:.9f}, ' +
                f'Accumulated Regularisation Loss: {accumulated_regularisation_loss:.9f}, ' +
                f'Learning Rate: {self.optimiser.get_current_learning_rate():.9f}'
            )

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

    def validate(self, x_val: np.ndarray, y_val: np.ndarray, batch_size: int = None) -> None:
        '''
        Perform model validation.
        '''
        # The default value if batch size is not set.
        steps: int = 1

        if batch_size is not None:
            steps = math.ceil(len(x_val) / batch_size)

        self.loss_function.reset_accumulation()
        self.accuracy.reset_accumulation()

        for step in range(steps):
            batch_x_val: np.ndarray = x_val
            batch_y_val: np.ndarray = y_val

            if batch_size is not None:
                batch_x_val = x_val[step *
                                    batch_size: (step + 1) * batch_size]
                batch_y_val = y_val[step *
                                    batch_size: (step + 1) * batch_size]

            output: np.ndarray = self.prediction(batch_x_val)

            # Calculate model loss.
            loss = self.loss_function.calculate(output, batch_y_val)

            # Get predictions.
            predictions = self.prediction_layer.prediction(output)

            # Calculate accuracy.
            accuracy: np.number = self.accuracy.calculate(
                predictions, batch_y_val, self.combined_function_type)

            if step <= steps:
                print(
                    f'Step: {step}, ' +
                    f'Validation Accuracy: {accuracy:.9f}, ' +
                    f'Validation Loss: {loss:.9f}'
                )

        if batch_size is None:
            return

        # Calculate the accumulated loss.
        accumulated_loss: np.number = self.loss_function.calculate_accumulated_loss()

        # Calculate the accumulated accuracy.
        accumulated_accuracy: np.number = self.accuracy.calculate_accumulated_accuracy()

        print(
            f'Accumulated Accuracy: {accumulated_accuracy:.3f}, ' +
            f'Accumulated Loss: {accumulated_loss:.9f}'
        )
