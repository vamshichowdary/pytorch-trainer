import numpy as np
from .metric import Metric


class L1Error(Metric):
    """L1 error.

    Args:
        per_pixel (boolean, optional): whether to normalize the error with number of pixels. Default: True.
        ignore_value (None or tensor) : Pixels with value in target to ignore (useful for eg if some target pixels contain NaNs and we do not 
        want these pixels to contribute to the total error)
    """

    def __init__(self, per_pixel=True, ignore_value=None):
        super().__init__()
        
        self.reset()
        self.per_pixel = per_pixel
        self.ignore_value = ignore_value

    def reset(self):
        self.n = 0
        self.l1_sum = 0.

    def add(self, predicted, target):
        """Computes the total L1 error (Absolute error)
        Args:
            predicted (numpy.ndarray) : prediction from the model
            target (numpy.ndarray) : target output value
        """
        if self.ignore_value is not None:
            mask = np.not_equal(target, self.ignore_value)
            self.l1_sum += np.abs(mask*(predicted-target)).sum()
            self.n += mask.sum()
        else:
            self.l1_sum += np.abs(predicted-target).sum()
            self.n += target.size
        
    def value(self):
        """
        Returns:
            L1 error (or normalized per pixel error (MAE))
        """
        if self.per_pixel:
            return self.l1_sum / max(self.n, 1)
        else:
            return self.l1_sum
