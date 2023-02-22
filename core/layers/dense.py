from __future__ import annotations
import numpy as np

from .layer import Layer


class Dense(Layer):
    '''
    The dense layer.
    '''

    def __init__(self, n_inputs: int, n_neurons: int,
                 weight_regulariser_l1: np.number = 0, weight_regulariser_l2: np.number = 0,
                 bias_regulariser_l1: np.ndarray = 0, bias_regulariser_l2: np.number = 0) -> None:
        '''
        Initialise weights and biases.
        '''
        super().__init__(is_trainable=True)

        self.inputs: np.ndarray = []
        self.output: np.ndarray = []
        self.weights: np.ndarray = 0.01 * \
            np.random.randn(n_inputs, n_neurons)
        self.biases: np.ndarray = np.zeros((1, n_neurons))

        self.d_weights: np.ndarray = []
        self.d_biases: np.ndarray = []

        self.weight_momentums: np.ndarray = []
        self.bias_momentums: np.ndarray = []

        self.weights_cache: np.ndarray = []
        self.biases_cache: np.ndarray = []

        # Regularisation strength.
        self.weight_regulariser_l1 = weight_regulariser_l1
        self.weight_regulariser_l2 = weight_regulariser_l2
        self.bias_regulariser_l1 = bias_regulariser_l1
        self.bias_regulariser_l2 = bias_regulariser_l2

    def forward(self, inputs: np.ndarray) -> None:
        '''
        Perform a forward pass. Calculate output values from inputs, weights, and biases.
        '''
        self.inputs = inputs

        self.output = np.dot(inputs, self.weights) + self.biases

    def backward(self, d_values: np.ndarray) -> None:
        '''
        Perform a backward pass. Calculate the partial derivative of inputs, weights, and biases.
        '''
        # Gradients on parameters.
        self.d_weights = np.dot(self.inputs.T, d_values)
        self.d_biases = np.sum(d_values, axis=0, keepdims=True)

        # Gradients on regularisation.
        if self.weight_regulariser_l1 > 0:
            d_l1 = np.ones_like(self.weights)
            d_l1[self.weights < 0] = -1
            self.d_weights += self.weight_regulariser_l1 * d_l1

        if self.bias_regulariser_l1 > 0:
            d_l1 = np.ones_like(self.biases)
            d_l1[self.biases < 0] = -1
            self.d_biases += self.bias_regulariser_l1 * d_l1

        if self.weight_regulariser_l2 > 0:
            self.d_weights += 2 * self.weight_regulariser_l2 * self.weights

        if self.bias_regulariser_l2 > 0:
            self.d_biases += 2 * self.bias_regulariser_l2 * self.biases

        # Gradients on values.
        super().set_d_inputs(np.dot(d_values, self.weights.T))

    def get_output(self) -> np.ndarray:
        '''
        Get outputs.
        '''
        return self.output

    def get_weights(self) -> np.ndarray:
        '''
        Get weights.
        '''
        return self.weights

    def set_weights(self, weights: np.ndarray) -> None:
        '''
        Set weights.
        '''
        self.weights = weights

    def get_biases(self) -> np.ndarray:
        '''
        Get biases.
        '''
        return self.biases

    def set_biases(self, biases: np.ndarray) -> None:
        '''
        Set biases.
        '''
        self.biases = biases

    def get_weights_cache(self) -> np.ndarray:
        '''
        Get weights cache.
        '''
        return self.weights_cache

    def set_weights_cache(self, weights: np.ndarray) -> None:
        '''
        Set weights cache.
        '''
        self.weights_cache = weights

    def get_biases_cache(self) -> np.ndarray:
        '''
        Get biases cache.
        '''
        return self.biases_cache

    def set_biases_cache(self, biases: np.ndarray) -> None:
        '''
        Set biases cache.
        '''
        self.biases_cache = biases

    def get_d_weights(self) -> np.ndarray:
        '''
        Get d_weights.
        '''
        return self.d_weights

    def get_d_biases(self) -> np.ndarray:
        '''
        Get d_biases.
        '''
        return self.d_biases

    def get_weight_momentums(self) -> np.ndarray:
        '''
        Get weight momentums.
        '''
        return self.weight_momentums

    def set_weight_momentums(self, weight_momentums: np.ndarray) -> None:
        '''
        Set weight momentums.
        '''
        self.weight_momentums = weight_momentums

    def get_bias_momentums(self) -> np.ndarray:
        '''
        Get bias momentums.
        '''
        return self.bias_momentums

    def set_bias_momentums(self, bias_momentums: np.ndarray) -> None:
        '''
        Set bias momentums.
        '''
        self.bias_momentums = bias_momentums

    def get_weight_regulariser_l1(self) -> np.number:
        '''
        Get L1 weight regularisation.
        '''
        return self.weight_regulariser_l1

    def set_weight_regulariser_l1(self, weight_regulariser_l1: np.number) -> None:
        '''
        Set L1 weight regularisation.
        '''
        self.weight_regulariser_l1 = weight_regulariser_l1

    def get_weight_regulariser_l2(self) -> np.number:
        '''
        Get L2 weight regularisation.
        '''
        return self.weight_regulariser_l2

    def set_weight_regulariser_l2(self, weight_regulariser_l2: np.number) -> None:
        '''
        Set L2 weight regularisation.
        '''
        self.weight_regulariser_l2 = weight_regulariser_l2

    def get_bias_regulariser_l1(self) -> np.number:
        '''
        Get L1 bias regularisation.
        '''
        return self.bias_regulariser_l1

    def set_bias_regulariser_l1(self, bias_regulariser_l1: np.number) -> None:
        '''
        Set L1 bias regularisation.
        '''
        self.bias_regulariser_l1 = bias_regulariser_l1

    def get_bias_regulariser_l2(self) -> np.number:
        '''
        Get L2 bias regularisation.
        '''
        return self.bias_regulariser_l2

    def set_bias_regulariser_l2(self, bias_regulariser_l2: np.number) -> None:
        '''
        Set L2 bias regularisation.
        '''
        self.bias_regulariser_l2 = bias_regulariser_l2
