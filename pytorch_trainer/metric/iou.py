import numpy as np
from .metric import Metric
from .confusionmatrix import ConfusionMatrix


class IoU(Metric):
    """Computes the intersection over union (IoU) per class.

    Intersection over union (IoU) is a common evaluation metric for semantic
    segmentation. The predictions are first accumulated in a confusion matrix
    and the IoU is computed from it as follows:

        IoU = true_positive / (true_positive + false_positive + false_negative).

    Keyword arguments:
    - num_classes (int): number of classes in the classification problem
    - normalized (boolean, optional): Determines whether or not the confusion
    matrix is normalized or not. Default: False.
    - ignore_index (int or iterable, optional): Index of the classes to ignore
    when computing the IoU. Can be an int, or any iterable of ints.
    """

    def __init__(self, num_classes, normalized=False, ignore_index=None):
        super().__init__()
        self.conf_metric = ConfusionMatrix(num_classes, normalized)

        if ignore_index is None:
            self.ignore_index = None
        elif isinstance(ignore_index, int):
            self.ignore_index = (ignore_index,)
        else:
            try:
                self.ignore_index = tuple(ignore_index)
            except TypeError:
                raise ValueError("'ignore_index' must be an int or iterable")

    def reset(self):
        self.conf_metric.reset()

    def add(self, predicted, target):
        """Adds the predicted and target pair to the IoU metric.

        Keyword arguments:
        - predicted (numpy.ndarray): Can be a (N, K, H, W) array of
        predicted scores obtained from the model for N examples and K classes,
        or (N, H, W) array of integer values between 0 and K-1.
        - target (numpy.ndarray): Can be a (N, K, H, W) array of
        target scores for N examples and K classes, or (N, H, W) array of
        integer values between 0 and K-1.

        """
        # Dimensions check
        assert predicted.shape[0] == target.shape[0], \
            'number of targets and predicted outputs do not match'
        assert predicted.ndim == 3 or predicted.ndim == 4, \
            "predictions must be of dimension (N, H, W) or (N, K, H, W)"
        assert target.ndim == 3 or target.ndim == 4, \
            "targets must be of dimension (N, H, W) or (N, K, H, W)"

        # If the array is in categorical format convert it to integer format
        if predicted.ndim == 4:
            predicted = predicted.argmax(1)
        if target.ndim == 4:
            target = target.argmax(1)

        self.conf_metric.add(predicted.reshape(-1), target.reshape(-1))

    def value(self):
        """Computes the IoU per class.
        Returns:
            IoU(List): per class IoU, for K classes it's a list with K elements.
        """
        conf_matrix = self.conf_metric.value()
        if self.ignore_index is not None:
            conf_matrix[:, self.ignore_index] = 0
            conf_matrix[self.ignore_index, :] = 0
        true_positive = np.diag(conf_matrix)
        false_positive = np.sum(conf_matrix, 0) - true_positive
        false_negative = np.sum(conf_matrix, 1) - true_positive

        # Just in case we get a division by 0, ignore/hide the error
        with np.errstate(divide='ignore', invalid='ignore'):
            iou = true_positive / (true_positive + false_positive + false_negative)

        return list(iou)

class MeanIOU(Metric):
    """
    Computes mean (mIoU)
    Keyword arguments:
    - num_classes (int): number of classes in the classification problem
    - normalized (boolean, optional): Determines whether or not the confusion
    matrix is normalized or not. Default: False.
    - ignore_index (int or iterable, optional): Index of the classes to ignore
    when computing the IoU. Can be an int, or any iterable of ints.
    """

    def __init__(self, num_classes, normalized=False, ignore_index=None):
        super().__init__()
        self.iou = IoU(num_classes, normalized, ignore_index)

    def reset(self):
        self.iou.reset()

    def add(self, predicted, target):
        """Adds the predicted and target pair to the IoU metric.

        Keyword arguments:
        - predicted (numpy.ndarray): Can be a (N, K, H, W) array of
        predicted scores obtained from the model for N examples and K classes,
        or (N, H, W) array of integer values between 0 and K-1.
        - target (numpy.ndarray): Can be a (N, K, H, W) array of
        target scores for N examples and K classes, or (N, H, W) array of
        integer values between 0 and K-1.

        """
        self.iou.add(predicted, target)

    def value(self):
        """Computes mean IoU.

        The mean computation ignores NaN elements of the IoU array.

        Returns:
            mean IoU.
        """
        return np.nanmean(self.iou)