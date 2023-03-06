import numpy as np
from math import ceil


class DataLoader:
    '''
    The base data class.
    '''

    def __init__(self, x: np.ndarray, y: np.ndarray) -> None:
        '''
        '''
        # Calculate the training and validation dataset counts.
        train_count: int = ceil(len(x) * 0.8)
        val_count: int = len(x) - train_count

        self.x_train: np.ndarray = x[:train_count]
        self.y_train: np.ndarray = y[:train_count]
        self.x_val: np.ndarray = x[-val_count:]
        self.y_val: np.ndarray = y[-val_count:]

    def get_training_data(self) -> tuple[np.ndarray, np.ndarray]:
        '''
        Get the training data.
        '''
        return (self.x_train, self.y_train)

    def get_validation_data(self) -> tuple[np.ndarray, np.ndarray]:
        '''
        Get the validation data.
        '''
        return (self.x_val, self.y_val)
