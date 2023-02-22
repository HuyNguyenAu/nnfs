import numpy as np

from ..layers import Dense
from .optimiser import Optimiser


class SGD(Optimiser):
    '''
    The Stocastic Gradient Descent optimiser.
    '''

    def __init__(self, learning_rate: np.number = 1.0, decay: np.number = 0.0,
                 momentum: np.number = 0.0) -> None:
        '''
        Initialise optimiser.
        Default learning rate of 1.
        '''
        super().__init__(learning_rate, decay)
        self.momentum: np.number = momentum

    def update_parameters(self, layer: Dense):
        '''
        Update parameters.
        '''
        weight_updates: np.ndarray = []
        bias_updates: np.ndarray = []

        if self.momentum:
            # Create an array of zero valuen momentums if it does not exist.
            if len(layer.get_weight_momentums()) <= 0:
                layer.set_weight_momentums(np.zeros_like(layer.get_weights()))

                # We're assuming if there is no weight momentums, then there are also
                # no bias momentums.
                layer.set_bias_momentums(np.zeros_like(layer.get_biases()))

            # Build weight updates with momentum by taking the previous updates multiplied
            # by the retain factor and update with current gradients.
            weight_updates = self.momentum * layer.get_weight_momentums() - \
                self.get_current_learning_rate() * layer.get_d_weights()

            layer.set_weight_momentums(weight_updates)

            # Build bias updates.
            bias_updates = self.momentum * layer.get_bias_momentums() - \
                self.get_current_learning_rate() * layer.get_d_biases()

            layer.set_bias_momentums(bias_updates)
        else:
            # Vanilla SGD updates.
            weight_updates = -self.get_current_learning_rate() * layer.get_d_weights()
            bias_updates = -self.get_current_learning_rate() * layer.get_d_biases()

        layer.set_weights(layer.get_weights() + weight_updates)
        layer.set_biases(layer.get_biases() + bias_updates)
