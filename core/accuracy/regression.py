import numpy as np

from .accuracy import Accuracy


class Regression(Accuracy):
    '''
    The regression accuracy class.
    '''

    def __init__(self) -> None:
        '''
        '''
        self.precision: float = None

    def init(self, y: np.ndarray, re_init: bool = False) -> None:
        '''
        Calculate the precision value based on the passed-in ground truth values.
        '''
        if self.precision is None or re_init:
            self.precision = np.std(y) / 250

    def compare(self, predictions: np.ndarray, y: np.ndarray) -> bool:
        return np.absolute(predictions - y) < self.precision
