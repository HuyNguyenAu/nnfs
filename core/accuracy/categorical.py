import numpy as np

from .accuracy import Accuracy


class Categorical(Accuracy):
    '''
    The categorical accuracy class.
    '''

    def __init__(self, *, binary=False) -> None:
        '''
        '''
        self.binary: float = binary

    def compare(self, *, predictions: np.ndarray, y: np.ndarray) -> bool:
        # If labels are one-hot encoded, turn them into discrete values.
        # Eg. y_true = [[1, 0, 0], [0, 1, 0], ...]
        if not self.binary and len(y.shape) == 2:
            # Get the array of indexes for the position on the '1's.
            #  Eg. y_true = np.argmax => [0, 1, ...]
            y = np.argmax(y, axis=1)

        return predictions == y
