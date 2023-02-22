import numpy as np

from .activation_function import ActivationFunction


class ReLU(ActivationFunction):
    '''
    ReLu activation function.
    '''

    def forward(self, inputs: np.ndarray) -> None:
        self.set_inputs(inputs)
        self.set_output(np.maximum(0, inputs))

    def backward(self, d_values: np.ndarray) -> None:
        # Set negative gradients to zero.
        d_inputs = d_values.copy()
        d_inputs[self.get_inputs() <= 0] = 0

        self.set_d_inputs(d_inputs)

    def prediction(self, output: np.ndarray) -> np.ndarray:
        return output
