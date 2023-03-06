from __future__ import annotations
import numpy as np

from ..types import CombinedFunctionType


class Accuracy:
    '''
    The base accuracy class.
    '''

    def __init__(self) -> None:
        '''
        '''
        self.accumulated_comparision: float = 0
        self.accumulated_sample_count: int = 0

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

        comparision: np.ndarray = self.compare(predictions=predictions, y=y)

        self.accumulated_comparision += np.sum(comparision)
        self.accumulated_sample_count += len(comparision)

        return np.mean(comparision)

    def calculate_accumulated_accuracy(self) -> np.ndarray:
        '''
        Calculate the accumulated accuracy.
        '''
        return self.accumulated_comparision / self.accumulated_sample_count

    def reset_accumulation(self) -> None:
        '''
        Set the accumulated loss and sample count to zero.
        '''
        self.accumulated_loss = 0
        self.accumulated_sample_count = 0
