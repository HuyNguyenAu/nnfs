from __future__ import annotations
import numpy as np


class Accuracy:
    '''
    The base accuracy class.
    '''

    def init(self, y: np.ndarray, re_init: bool = False) -> None:
        '''
        Calculate the precision value based on the passed-in ground truth values.
        '''
        return NotImplementedError

    def compare(self, predictions: np.ndarray, y: np.ndarray) -> bool:
        '''
        Compares the predictions to the ground truth values.
        '''
        return NotImplementedError

    def calculate(self, predictions: np.ndarray, y: np.ndarray) -> np.float32:
        '''
        Calculate the accuracy of the given predictions and ground truths.
        '''
        return np.mean(self.compare(predictions, y))
