from __future__ import annotations
import numpy as np

from ..layers import Layer


class ActivationFunction(Layer):
    '''
    The activation function base class.
    '''

    def __init__(self) -> None:
        '''
        '''
        super().__init__(is_trainable=True)

        self.inputs: np.ndarray = []
        self.d_inputs: np.ndarray = []

    def forward(self, inputs: np.ndarray) -> None:
        '''
        Perform a forward pass.
        '''
        raise NotImplementedError

    def backward(self, d_values: np.ndarray) -> None:
        '''
        Perfom a backwards pass.
        '''
        raise NotImplementedError

    def prediction(self, output: np.ndarray) -> np.ndarray:
        '''
        Calculate the predictions for the output.
        '''
        raise NotImplementedError

    def get_inputs(self) -> np.ndarray:
        '''
        Get inputs.
        '''
        return self.inputs

    def set_inputs(self, inputs: np.ndarray) -> None:
        '''
        Set inputs.
        '''
        self.inputs = inputs

    def get_d_inputs(self) -> np.ndarray:
        '''
        Get d_inputs.
        '''
        return self.d_inputs

    def set_d_inputs(self, d_inputs: np.ndarray) -> None:
        '''
        Set d_inputs.
        '''
        self.d_inputs = d_inputs

    def update_d_inputs(self, index: int, d_input: any) -> None:
        '''
        Update a single d_inputs value at given index.
        '''
        self.d_inputs[index] = d_input
