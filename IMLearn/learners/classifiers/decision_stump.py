from __future__ import annotations
from typing import Tuple, NoReturn
from ...base import BaseEstimator, BaseModule
import numpy as np
from itertools import product
from ...metrics import misclassification_error


class DecisionStump(BaseEstimator):
    """
    A decision stump classifier for {-1,1} labels according to the CART algorithm

    Attributes
    ----------
    self.threshold_ : float
        The threshold by which the data is split

    self.j_ : int
        The index of the feature by which to split the data

    self.sign_: int
        The label to predict for samples where the value of the j'th feature is about the threshold
    """

    def __init__(self) -> DecisionStump:
        """
        Instantiate a Decision stump classifier
        """
        super().__init__()
        self.threshold_, self.j_, self.sign_ = None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits a decision stump to the given data

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        best_threshold, best_error, best_j, best_sign = None, np.inf, None, None
        for j in range(X.shape[1]):
            theshold, error = self._find_threshold(X[:, j], y, 1)
            if error < best_error:
                best_threshold, best_error, best_j, best_sign = theshold, error, j, 1
            theshold, error = self._find_threshold(X[:, j], y, -1)
            if error < best_error:
                best_threshold, best_error, best_j, best_sign = theshold, error, j, -1
        self.threshold_, self.j_, self.sign_ = best_threshold, best_j, best_sign

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples

        Notes
        -----
        Feature values strictly below threshold are predicted as `-sign` whereas values which equal
        to or above the threshold are predicted as `sign`
        """
        return ((X[:, self.j_] >= self.threshold_) * self.sign_) * 2 - self.sign_

    def _find_threshold(self, values: np.ndarray, labels: np.ndarray, sign: int) -> Tuple[float, float]:
        """
        Given a feature vector and labels, find a threshold by which to perform a split
        The threshold is found according to the value minimizing the misclassification
        error along this feature

        Parameters
        ----------
        values: ndarray of shape (n_samples,)
            A feature vector to find a splitting threshold for

        labels: ndarray of shape (n_samples,)
            The labels to compare against

        sign: int
            Predicted label assigned to values equal to or above threshold

        Returns
        -------
        thr: float
            Threshold by which to perform split

        thr_err: float between 0 and 1
            Misclassificaiton error of returned threshold

        Notes
        -----
        For every tested threshold, values strictly below threshold are predicted as `-sign` whereas values
        which equal to or above the threshold are predicted as `sign`
        """
        values_labels = np.vstack((values, labels)).T  # [[value_1, label_1], [value_2, label_2], ...]
        # Find the best threshold by minimizing the misclassification error
        thresholds = np.unique(values_labels[:, 0])
        errors = np.zeros(len(thresholds))
        for i, threshold in enumerate(thresholds):
            new_value_labels = values_labels
            new_value_labels[new_value_labels[:, 0] >= threshold, 1] = sign
            new_value_labels[new_value_labels[:, 0] < threshold, 1] = -sign
            errors[i] = np.sum(np.abs(labels[new_value_labels[:, 1] != np.sign(labels)]))  # weighted misclassification error
        return thresholds[np.argmin(errors)], np.min(errors)/len(labels)

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under misclassification loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        Returns
        -------
        loss : float
            Performance under missclassification loss function
        """
        return misclassification_error(self.predict(X), y)


# if __name__ == "__main__":
#     a = np.array([1, 2, 3, 4, 2, 1, 2, 3, 5, 6]).reshape(10, 1)
#     y = np.array([-1 if i < 3 else 1 for i in a])
#     s = DecisionStump()
#     s.fit(a, y)
#     print(s.predict(a))
