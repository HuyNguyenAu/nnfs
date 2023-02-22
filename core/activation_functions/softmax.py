import numpy as np

from .activation_function import ActivationFunction


class Softmax(ActivationFunction):
    '''
    The Softmax activation function.
    '''

    def forward(self, inputs: np.ndarray) -> None:
        # Get unnormalised probabilities.
        unnormalised_values: np.ndarray = np.exp(
            inputs - np.max(inputs, axis=1, keepdims=True))
        # Normalise each sample.
        probabilities: np.ndarray = unnormalised_values / \
            np.sum(unnormalised_values, axis=1, keepdims=True)

        self.set_output(probabilities)

    def backward(self, d_values: np.ndarray) -> None:
        self.set_d_inputs(np.empty_like(d_values))

        # Enumerate outputs and gradients.
        for index, (single_output, single_d_values) in enumerate(zip(self.get_output(), d_values)):
            # Faltten output array.
            single_output = single_output.reshape(-1, 1)

            # Calculate the Jacobian matrix from the output.
            jacobian_matrix = np.diagflat(
                single_output) - np.dot(single_output, single_output.T)

            # Calculate the sample-wise gradient and add it to the sample gradients.
            self.update_d_inputs(index, np.dot(
                jacobian_matrix, single_d_values))

    def prediction(self, output: np.ndarray) -> np.ndarray:
        return np.argmax(output, axis=1)
