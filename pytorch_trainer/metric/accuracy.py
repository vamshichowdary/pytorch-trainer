import numpy as np
from .metric import Metric

class Accuracy(Metric):
    """ Accuracy.

    Args:
        per_pixel (boolean, optional): whether to normalize the error with number of pixels. Default: True.
    """

    def __init__(self):
        super().__init__()
        self.reset()

    def reset(self):
        self.correct = 0
        self.total = 0

    def add(self, predicted, target):
        """
        Args:
            predicted (numpy.ndarray) : prediction from the model
            target (numpy.ndarray) : target output value
        """
        predicted = np.argmax(predicted, axis=1)
        self.total += target.shape[0]
        self.correct += (predicted == target).sum().item()
        
    def value(self):
        """
        Returns:
            Accuracy as a percentage
        """
        return 100 * self.correct/max(self.total, 1)
