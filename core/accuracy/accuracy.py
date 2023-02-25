from __future__ import annotations
import numpy as np

from ..types import CombinedFunctionType


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

    def calculate(self, predictions: np.ndarray, y: np.ndarray, combined_function_type: CombinedFunctionType) -> np.float32:
        '''
        Calculate the accuracy of the given predictions and ground truths.
        '''
        if combined_function_type == CombinedFunctionType.ACTIVATION_LOSS:
            # If labels are one-hot encoded, turn them into discrete values.
            if len(y.shape) == 2:
                y = np.argmax(y, axis=1)

        return np.mean(self.compare(predictions, y))
