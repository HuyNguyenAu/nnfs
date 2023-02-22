from __future__ import annotations
import numpy as np

from ..layers import Dense


class Optimiser:
    '''
    The optimiser base class.
    '''

    def __init__(self, learning_rate: np.number = 1.0, decay: np.number = 0.0) -> None:
        '''
        Initialise optimiser.
        Default learning rate of 1.
        '''
        super().__init__()

        self.learning_rate: np.number = learning_rate
        self.current_learning_rate: np.number = learning_rate
        self.decay: np.number = decay
        self.iterations: np.number = 0

    def pre_update_parameters(self) -> None:
        '''
        Call only once before any parameter updates.
        '''
        if self.decay:
            self.current_learning_rate = self.learning_rate * \
                (1.0 / (1.0 + self.decay * self.iterations))

    def update_parameters(self, layer: Dense):
        '''
        Update parameters.
        '''
        raise NotImplementedError

    def post_update_parameters(self):
        '''
        Call only once after any parameter updates
        '''
        self.iterations += 1

    def get_current_learning_rate(self):
        '''
        Get current learning rate.
        '''
        return self.current_learning_rate

    def get_iterations(self):
        '''
        Get iterations.
        '''
        return self.iterations
