import numpy as np

from .activation_function import ActivationFunction


class Sigmoid(ActivationFunction):
    '''
    The Sigmoid activation function.
    '''

    def forward(self, inputs: np.ndarray):
        self.set_inputs(inputs)
        self.set_output(1 / (1 + np.exp(-inputs)))

    def backward(self, d_values: np.ndarray):
        self.set_d_inputs(d_values * (1 - self.get_output())
                          * self.get_output())

    def prediction(self, output: np.ndarray) -> np.ndarray:
        return (output > 0.5) * 1.0
