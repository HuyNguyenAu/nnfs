import numpy as np

from ..layers import Dense
from .optimiser import Optimiser


class AdaGrad(Optimiser):
    '''
    The adaptive gradient optimiser.
    '''

    def __init__(self, learning_rate: np.number = 1.0,
                 decay: np.number = 0.0, epsilon: np.number = 1e-7) -> None:
        '''
        Initialise optimiser.
        Default learning rate of 1.
        '''
        super().__init__(
            learning_rate, decay,
        )
        self.epsilon: np.number = epsilon

    def update_parameters(self, layer: Dense):
        '''
        Update parameters.
        '''
        # Create an array of zero valuen momentums if it does not exist.
        if len(layer.get_weights_cache()) <= 0:
            layer.set_weights_cache(np.zeros_like(layer.get_weights()))
            layer.set_biases_cache(np.zeros_like(layer.get_biases()))

        # Update cache.
        layer.set_weights_cache(
            layer.get_weights_cache() + (layer.get_d_weights() ** 2))
        layer.set_biases_cache(layer.get_biases_cache() +
                               (layer.get_d_biases() ** 2))

        # Vanilla SGD parameters with squared current gradients.
        weights = -self.get_current_learning_rate() * layer.get_d_weights() / \
            (np.sqrt(layer.get_weights_cache()) + self.epsilon)
        layer.set_weights(layer.get_weights() + weights)

        biases = -self.get_current_learning_rate() * layer.get_d_biases() / \
            (np.sqrt(layer.get_biases_cache()) + self.epsilon)
        layer.set_biases(layer.get_biases() + biases)
