import numpy as np
from .activation_function import ActivationFunction


class Linear(ActivationFunction):
    '''
    The linear activation function.
    '''

    def forward(self, inputs: np.ndarray) -> None:
        self.set_inputs(inputs)
        self.set_output(inputs)

    def backward(self, d_values: np.ndarray) -> None:
        self.set_d_inputs(d_values.copy())

    def prediction(self, output: np.ndarray) -> np.ndarray:
        return output
