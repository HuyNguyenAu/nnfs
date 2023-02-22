from __future__ import annotations
from uuid import UUID, uuid4
import numpy as np


class Layer:
    '''
    The layer base class.
    '''

    def __init__(self, *, is_trainable: bool = False) -> None:
        '''
        '''
        self.id: UUID = uuid4()
        self.d_inputs: np.ndarray = []
        self.output: np.ndarray = []
        self.previous_layer: UUID = None
        self.next_layer: UUID = None
        self.is_trainable: bool = is_trainable
        self.is_disabled: bool = False

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

    def get_id(self) -> UUID:
        '''
        Get the layer id.
        '''
        return self.id

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

    def get_output(self) -> np.ndarray:
        '''
        Get output.
        '''
        return self.output

    def set_output(self, output: np.ndarray) -> None:
        '''
        Set output.
        '''
        self.output = output

    def get_previous_layer(self) -> UUID:
        '''
        Get the previous layer id.
        '''
        return self.previous_layer

    def set_previous_layer(self, layer_id: UUID) -> None:
        '''
        Set the previous layer id.
        '''
        self.previous_layer = layer_id

    def get_next_layer(self) -> UUID:
        '''
        Get the next layer id.
        '''
        return self.next_layer

    def set_next_layer(self, layer_id: UUID) -> None:
        '''
        Set the next layer id.
        '''
        self.next_layer = layer_id

    def get_is_trainable(self) -> bool:
        '''
        Indicates if the layer is trainable.
        '''
        return self.is_trainable

    def get_is_disabled(self) -> bool:
        '''
        Indicates if the layer is disabled.
        '''
        return self.is_disabled

    def set_is_disabled(self, disabled: bool) -> None:
        '''
        Set if the layer is disabled.
        '''
        self.is_disabled = disabled
