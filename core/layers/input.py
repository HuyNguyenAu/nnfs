import numpy as np

from .layer import Layer


class Input(Layer):
    '''
    The input layer.
    '''

    def __init__(self) -> None:
        '''
        '''
        super().__init__(is_trainable=True)

    def forward(self, inputs: np.ndarray) -> None:
        self.set_output(inputs)
