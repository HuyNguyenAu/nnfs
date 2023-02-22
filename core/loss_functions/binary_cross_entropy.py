import numpy as np
from .loss_function import LossFunction


class BinaryCrossEntropy(LossFunction):
    '''
    The Binary Cross Entropy loss function.
    '''

    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        '''
        Forward pass.
        '''
        # Clip the y_pred to prevent division by zero.
        # Clipped both sides to prevent dragging mean to any value.
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        # Calculate sample-wise loss.
        sample_losses = -(y_true * np.log(y_pred_clipped) +
                          (1 - y_true) * np.log(1 - y_pred_clipped))
        sample_losses = np.mean(sample_losses, axis=-1)

        return sample_losses

    def backward(self, d_values: np.ndarray, y_true: np.ndarray):
        '''
        Backwards pass.
        '''
        samples = len(d_values)
        # Use the first sample to count the number of outputs in every sample.
        outputs = len(d_values[0])

        clipped_d_values = np.clip(d_values, 1e-7, 1 - 1e-7)

        self.set_d_inputs(-(y_true / clipped_d_values -
                          (1 - y_true) / (1 - clipped_d_values)) / outputs)

        # Normalise the the gradient.
        self.set_d_inputs(self.get_d_inputs() / samples)
