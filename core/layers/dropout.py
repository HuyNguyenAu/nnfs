import numpy as np

from .layer import Layer


class Dropout(Layer):
    '''
    The dropout layer.
    '''

    def __init__(self, rate: np.number):
        '''
        Store the dropout rate.
        '''
        super().__init__(is_trainable=True)

        self.rate: np.number = 1 - rate
        self.binary_mask: np.ndarray = []
        self.inputs: np.ndarray = []

    def forward(self, inputs: np.ndarray):
        '''
        Forward pass.
        '''

        # Store the input values.
        self.inputs = inputs

        if super().get_is_disabled():
            super().set_output(inputs.copy())
            return

        # Generate and save scaled mask.
        self.binary_mask = np.random.binomial(
            1, self.rate, size=inputs.shape) / self.rate

        # Apply mask to output.
        super().set_output(inputs * self.binary_mask)

    def backward(self, d_values: np.ndarray):
        # Gradient on values.
        super().set_d_inputs(d_values * self.binary_mask)
