import numpy as np

from ..activation_functions import ActivationFunction
from ..loss_functions import CategoricalCrossEntropy
from ..activation_functions import Softmax


class SoftmaxCategoricalCrossEntropy(ActivationFunction):
    '''
    The Categorical Cross Entropy loss function combined with
    Softmax Activation for faster backwards pass.
    '''

    def __init__(self) -> None:
        '''
        '''
        self.activation: Softmax = Softmax()
        self.loss: CategoricalCrossEntropy = CategoricalCrossEntropy()
        self.output: np.ndarray = []

    def forward(self, inputs: np.ndarray, y_true: np.ndarray) -> None:
        '''
        Forward pass.
        '''
        self.activation.forward(inputs)

        self.set_output(self.activation.get_output())

        # Calculate the loss.
        return self.loss.calculate(self.output, y_true)

    def backward(self, d_values: np.ndarray, y_true: np.ndarray) -> None:
        '''
        Backward pass.
        '''
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
        self.set_d_inputs(d_inputs / samples)
