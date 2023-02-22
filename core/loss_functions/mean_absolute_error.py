import numpy as np

from .loss_function import LossFunction


class MeanAbsoluteError(LossFunction):
    '''
    The mean absolute error loss function.
    '''

    def __init__(self) -> None:
        '''
        '''
        pass

    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> None:
        return np.mean(np.abs(y_true - y_pred), axis=-1)

    def backward(self, d_values: np.ndarray, y_true: np.ndarray) -> None:
        self.set_d_inputs(np.sign(y_true - d_values) / len(d_values[0]))
        self.set_d_inputs(self.get_d_inputs() / len(d_values))
