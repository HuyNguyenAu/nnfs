import numpy as np

from .optimiser import Optimiser
from ..layers import Dense


class Adam(Optimiser):
    '''
    The Adaptive Momentum optimiser.
    '''

    def __init__(self, learning_rate: np.number = 0.001, decay: np.number = 0.0,
                 epsilon: np.number = 1e-7, beta_1: np.number = 0.9, beta_2: np.number = 0.999) -> None:
        '''
        Initialise optimiser.
        Default learning rate of 1.
        '''
        super().__init__(
            learning_rate, decay
        )
        self.epsilon: np.number = epsilon
        self.beta_1: np.number = beta_1
        self.beta_2: np.number = beta_2

    def update_parameters(self, layer: Dense):
        '''
        Update parameters.
        '''
        # Create an array of zero valuen momentums if it does not exist.
        if len(layer.get_weights_cache()) <= 0:
            layer.set_weight_momentums(np.zeros_like(layer.get_weights()))
            layer.set_weights_cache(np.zeros_like(layer.get_weights()))
            layer.set_bias_momentums(np.zeros_like(layer.get_biases()))
            layer.set_biases_cache(np.zeros_like(layer.get_biases()))

        # Update momentum with current gradients.
        layer.set_weight_momentums(
            self.beta_1 * layer.get_weight_momentums() + (1 - self.beta_1) * layer.get_d_weights())
        layer.set_bias_momentums(self.beta_1 * layer.get_bias_momentums() +
                                 (1 - self.beta_1) * layer.get_d_biases())

        # Get corrected momentum (self.iteration is a 0 with the first pass).
        # We need to start with 1 here.
        weight_momentums_corrected = layer.get_weight_momentums() / \
            (1 - self.beta_1 ** (self.get_iterations() + 1))
        bias_momentums_corrected = layer.get_bias_momentums() / \
            (1 - self.beta_1 ** (self.get_iterations() + 1))

        # Update the cache with squared current gradients.
        layer.set_weights_cache(self.beta_2 * layer.get_weights_cache() +
                                (1 - self.beta_2) * layer.get_d_weights() ** 2)
        layer.set_biases_cache(self.beta_2 * layer.get_biases_cache() +
                               (1 - self.beta_2) * layer.get_d_biases() ** 2)

        # Get corrected cache.
        weight_cache_corrected = layer.get_weights_cache() / \
            (1 - self.beta_2 ** (self.get_iterations() + 1))
        bias_cache_corrected = layer.get_biases_cache() / \
            (1 - self.beta_2 ** (self.get_iterations() + 1))

        # Vanilla SGD parameters update + normalisation with squared root cache.
        weights = -self.get_current_learning_rate() * weight_momentums_corrected / \
            (np.sqrt(weight_cache_corrected) + self.epsilon)
        layer.set_weights(layer.get_weights() + weights)

        biases = -self.get_current_learning_rate() * bias_momentums_corrected / \
            (np.sqrt(bias_cache_corrected) + self.epsilon)
        layer.set_biases(layer.get_biases() + biases)
