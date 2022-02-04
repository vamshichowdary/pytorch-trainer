import numpy as np
from .metric import Metric

class MeanLInfError(Metric):
    """
    L-infinity norm averaged across batched samples.
    """

    def __init__(self):
        super().__init__()
        self.reset()

    def reset(self):
        self.max_errs = np.empty(0)

    def add(self, predicted, target):
        """computes L-infinity norm for each mini batch sample.
        Args:
            predicted (numpy.ndarray) : prediction from the model
            target (numpy.ndarray) : target output value
        """
        errs = np.abs(predicted.reshape(predicted.shape[0], -1) - target.reshape(target.shape[0], -1)).max(axis=1)
        self.max_errs = np.concatenate([self.max_errs, errs])
        
    def value(self):
        """
        Returns:
            Averaged L-infinity norm for all samples
        """
        return self.max_errs.mean()