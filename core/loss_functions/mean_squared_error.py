import numpy as np

from .loss_function import LossFunction


class MeanSquaredError(LossFunction):
    '''
    The mean squared error loss function.
    '''

    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> None:
        return np.mean((y_true - y_pred) ** 2, axis=-1)

    def backward(self, d_values: np.ndarray, y_true: np.ndarray) -> None:
        self.set_d_inputs(-2 * (y_true - d_values) / len(d_values[0]))
        self.set_d_inputs(self.get_d_inputs() / len(d_values))
