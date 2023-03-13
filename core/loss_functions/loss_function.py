import numpy as np

from ..layers import Layer, Dense


class LossFunction(Layer):
    '''
    The base loss function class.
    '''

    def __init__(self) -> None:
        '''
        '''
        super().__init__(is_trainable=False)
        self.accumulated_loss: float = 0
        self.accumulated_sample_count: int = 0

    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> None:
        '''
        Perform a forward pass.
        '''
        raise NotImplementedError

    def backward(self, d_values: np.ndarray, y_true: np.ndarray) -> None:
        '''
        Perfom a backwards pass.
        '''
        raise NotImplementedError

    def calculate(self, output: np.ndarray, y: np.ndarray) -> np.ndarray:
        '''
        Calculate the data and regularlisation losses given the model output
        and ground truth values.
        '''
        sample_losses: np.ndarray = self.forward(output, y)
        data_loss: np.ndarray = np.mean(sample_losses)

        self.accumulated_loss += np.sum(sample_losses)
        self.accumulated_sample_count += len(sample_losses)

        return data_loss

    def calculate_accumulated_loss(self) -> np.ndarray:
        '''
        Calculate the accumulated loss.
        '''
        return self.accumulated_loss / self.accumulated_sample_count

    def regularisation_loss(self, layer: Dense) -> np.number:
        '''
        Calculate the regularisation loss.
        '''
        regularisation_loss = 0

        # Calculate the L1 weight regularisation only when it's larger than zero.
        if layer.get_weight_regulariser_l1() > 0:
            regularisation_loss += layer.get_weight_regulariser_l1() * \
                np.sum(np.abs(layer.get_weights()))

        # Calculate the L2 weight regularisation only when it's larger than zero.
        if layer.get_weight_regulariser_l2() > 0:
            regularisation_loss += layer.get_weight_regulariser_l2() * \
                np.sum(layer.get_weights() * layer.get_weights())

        # Calculate the L1 bias regularisation only when it's larger than zero.
        if layer.get_bias_regulariser_l1() > 0:
            regularisation_loss += layer.get_bias_regulariser_l1() * \
                np.sum(np.abs(layer.get_biases()))

        # Calculate the L2 bias regularisation only when it's larger than zero.
        if layer.get_bias_regulariser_l2() > 0:
            regularisation_loss += layer.get_bias_regulariser_l2() * \
                np.sum(layer.get_biases() * layer.get_biases())

        return regularisation_loss

    def reset_accumulation(self) -> None:
        '''
        Set the accumulated loss and sample count to zero.
        '''
        self.accumulated_loss = 0
        self.accumulated_sample_count = 0
