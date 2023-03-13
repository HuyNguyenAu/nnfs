from math import ceil
import numpy as np


class DataLoader:
    '''
    The base data class.
    '''

    def __init__(self, x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray = None, y_val: np.ndarray = None) -> None:
        '''
        '''
        self.x_train: np.ndarray = x_train
        self.y_train: np.ndarray = y_train
        self.x_val: np.ndarray = x_val
        self.y_val: np.ndarray = y_val

        # If no validation data is provided, continue to split the training data into
        # training and validation data.
        if x_val is not None:
            return

        # Calculate the training and validation dataset counts.
        train_count: int = ceil(len(x_train) * 0.8)
        val_count: int = len(x_train) - train_count

        self.x_train = x_train[:train_count]
        self.y_train = y_train[:train_count]
        self.x_val = x_train[-val_count:]
        self.y_val = y_train[-val_count:]
        
    def shuffle(self) -> None:
        '''
        Shuffle the training dataset.
        '''
        keys: list[int] = np.array(range(self.x_train.shape[0]))
        np.random.shuffle(keys)
        
        self.x_train = self.x_train[keys]
        self.y_train = self.x_train[keys]

    def get_x_train(self) -> np.ndarray:
        '''
        Get the x training data.
        '''
        return self.x_train

    def get_y_train(self) -> np.ndarray:
        '''
        Get the y training data.
        '''
        return self.y_train

    def get_x_val(self) -> np.ndarray:
        '''
        Get the x validation data.
        '''
        return self.x_val

    def get_y_val(self) -> np.ndarray:
        '''
        Get the x validation data.
        '''
        return self.y_val

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
