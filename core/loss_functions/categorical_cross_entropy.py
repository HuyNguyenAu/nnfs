import numpy as np

from .loss_function import LossFunction


class CategoricalCrossEntropy(LossFunction):
    '''
    The Categorical Cross Entropy loss function.
    '''

    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> None:
        '''
        Forward pass.
        '''
        samples: np.ndarray = len(y_pred)

        # Clip data to prevent ln(0). By clipping the upper and lower
        # bounds, we can avoid biasing the mean.
        y_pred_clipped: np.ndarray = np.clip(y_pred, 1e-7, 1 - 1e-7)

        # Handle sparse labels.
        # Sparse labels are values which we can't use to assign a class
        # to the input values.
        # In this case, that means that the output for each input value
        # is not a probability distribution for each class.
        if len(y_true.shape) == 1:
            # The targets are sparsed. The values represent the class targets,
            # which is used to show which class the input belongs to.
            # Eg. Input = [a], Targets: [cat: 0, dog: 1, other: 0]
            correct_confidences: np.ndarray = y_pred_clipped[range(
                samples), y_true]
        elif len(y_true.shape) == 2:
            # Eg. Input = [[a, b, c], ...], Targets: [[0.5, 0.4, 0.1], [0.5, 0.4, 0.1], [0.5, 0.4, 0.1]].
            correct_confidences: np.ndarray = np.sum(
                y_pred_clipped * y_true, axis=1)

        # Calculate losses.
        losses: np.ndarray = -np.log(correct_confidences)

        return losses

    def backward(self, d_values: np.ndarray, y_true: np.ndarray) -> None:
        '''
        Backward pass.
        '''
        samples: int = len(d_values)
        labels: int = len(d_values[0])

        # If labels are sparse, turn them into one-hot vector.
        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]

        # Calculate and normalise gradient.
        self.set_d_inputs((-y_true / d_values) / samples)
