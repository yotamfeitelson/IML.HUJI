from typing import NoReturn
from ...base import BaseEstimator
import numpy as np
from numpy.linalg import det, inv
from ...metrics import misclassification_error
from scipy.stats import multivariate_normal
from ...learners import gaussian_estimators
def calc_pdf(sample, mu, cov):
    d = cov.shape[0]
    # manually calculate multivatical normal pdf
    pdf = 1 / (np.sqrt((2 * np.pi) ** d * det(cov))) * np.exp(-0.5 * np.dot(np.dot((sample - mu).T, inv(cov)), (sample - mu)))
    return pdf

class LDA(BaseEstimator):
    """
    Linear Discriminant Analysis (LDA) classifier

    Attributes
    ----------
    self.classes_ : np.ndarray of shape (n_classes,)
        The different labels classes. To be set in `LDA.fit`

    self.mu_ : np.ndarray of shape (n_classes,n_features)
        The estimated features means for each class. To be set in `LDA.fit`

    self.cov_ : np.ndarray of shape (n_features,n_features)
        The estimated features covariance. To be set in `LDA.fit`

    self._cov_inv : np.ndarray of shape (n_features,n_features)
        The inverse of the estimated features covariance. To be set in `LDA.fit`

    self.pi_: np.ndarray of shape (n_classes)
        The estimated class probabilities. To be set in `LDA.fit`
    """

    def __init__(self):
        """
        Instantiate an LDA classifier
        """
        super().__init__()
        self.classes_, self.mu_, self.cov_, self._cov_inv, self.pi_ = None, None, None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits an LDA model.
        Estimates gaussian for each label class - Different mean vector, same covariance
        matrix with dependent features.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        self.classes_ = np.unique(y)  # y = [a,b,a,a,b,c,d,a,b,b] => classes_ = [a,b,c,d]
        self.mu_ = np.zeros((self.classes_.size, X.shape[1]))
        self.cov_ = np.zeros((X.shape[1], X.shape[1]))
        self._cov_inv = np.zeros((X.shape[1], X.shape[1]))
        self.pi_ = np.zeros(self.classes_.size)
        # Calculate the mean for each class
        for i, c in enumerate(self.classes_):
            self.mu_[i] = np.mean(X[y == c], axis=0)
            self.pi_[i] = np.sum(y == c) / y.size  # Probability of class c
        # Calculate the covariance matrix for each class
        for i, c in enumerate(self.classes_):
            self.cov_ += self.pi_[i] * np.cov(X[y == c].T, bias=self.classes_.size)
        # Calculate the inverse of the covariance matrix
        self._cov_inv = inv(self.cov_)
        self.fitted_ = True

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples
        """
        # # Predict the class for each sample
        # max_response = 0
        # max_class = 0
        # for i, c in enumerate(self.classes_):
        #     ak = self._cov_inv.dot(self.mu_[i])
        #     bk = np.log(self.pi_[i]) - 0.5 * self.mu_[i].dot(ak)
        #     response = ak.T.dot(X) + bk
        #     if response > max_response:
        #         max_response = response
        #         max_class = c
        # return max_class
        likelihoods = self.likelihood(X)
        return np.argmax(likelihoods, axis=1)

    def likelihood(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate the likelihood of a given data over the estimated model

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input data to calculate its likelihood over the different classes.

        Returns
        -------
        likelihoods : np.ndarray of shape (n_samples, n_classes)
            The likelihood for each sample under each of the classes
        """
        if not self.fitted_:
            raise ValueError("Estimator must first be fitted before calling `likelihood` function")
        likelihoods = np.zeros((X.shape[0], self.classes_.size))
        for i, c in enumerate(self.classes_):
            # likelihoods[:,i] = multivariate_normal.pdf(X, self.mu_[i], self.cov_) * self.pi_[i]
            pdfs = []
            for sample in X:
                pdfs.append(calc_pdf(sample, self.mu_[i], self.cov_))
            likelihoods[:, i] = np.array(pdfs) * self.pi_[i]

        return likelihoods

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
        if not self.fitted_:
            raise ValueError("Estimator must first be fitted before calling `loss` function")
        return misclassification_error(self.predict(X), y)
