import numpy as np

from core.layers.layer import Layer

from ..activation_functions import Softmax
from ..loss_functions import CategoricalCrossEntropy


class SoftmaxCategoricalCrossEntropy(Layer):
    '''
    The Categorical Cross Entropy loss function combined with
    Softmax Activation for faster backwards pass.
    '''

    def __init__(self) -> None:
        '''
        '''
        super().__init__()

        self.activation = Softmax()
        self.loss = CategoricalCrossEntropy()

    def forward(self, inputs: np.ndarray) -> None:
        '''
        Perform a forward pass.
        '''
        self.activation.forward(inputs)
        super().set_output(self.activation.get_output())

    def backward(self, d_values: np.ndarray, y_true: np.ndarray = None) -> None:
        '''
        Perform a backward pass.
        '''
        # This will keep the model backward pass simple, since it will only invoke
        # this function will a single parameter.
        if y_true is None:
            return
        
        samples: int = len(d_values)

        # If labels are one-hot encoded, turn them into discrete values.
        # Eg. y_true = [[1, 0, 0], [0, 1, 0], ...]
        if len(y_true.shape) == 2:
            # Get the array of indexes for the position on the '1's.
            #  Eg. y_true = np.argmax => [0, 1, ...]
            y_true = np.argmax(y_true, axis=1)

        d_inputs = d_values.copy()

        # Calculate gradients.
        d_inputs[range(samples), y_true] -= 1

        # Normalise gradients.
        super().set_d_inputs(d_inputs / samples)

    def calculate(self, output: np.ndarray, y: np.ndarray) -> np.ndarray:
        '''
        Calculate the data and regularlisation losses given the model output and ground truth values.
        '''
        return self.loss.calculate(output=output, y=y)
    
    def regularisation_loss(self, layer: any) -> np.number:
        '''
        Calculate the regularisation loss.
        '''
        return self.loss.regularisation_loss(layer)

    def prediction(self, output: np.ndarray) -> np.ndarray:
        '''
        Calculate the predictions for the output.
        '''
        return self.activation.prediction(output=output)
