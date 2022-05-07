from IMLearn.learners.classifiers import Perceptron, LDA, GaussianNaiveBayes
from typing import Tuple
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from math import atan2, pi


def load_dataset(filename: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load dataset for comparing the Gaussian Naive Bayes and LDA classifiers. File is assumed to be an
    ndarray of shape (n_samples, 3) where the first 2 columns represent features and the third column the class

    Parameters
    ----------
    filename: str
        Path to .npy data file

    Returns
    -------
    X: ndarray of shape (n_samples, 2)
        Design matrix to be used

    y: ndarray of shape (n_samples,)
        Class vector specifying for each sample its class

    """
    data = np.load(filename)
    return data[:, :2], data[:, 2].astype(int)


def run_perceptron():
    """
    Fit and plot fit progression of the Perceptron algorithm over both the linearly separable and inseparable datasets

    Create a line plot that shows the perceptron algorithm's training loss values (y-axis)
    as a function of the training iterations (x-axis).
    """
    fig = make_subplots(rows=2)
    row = 1
    for n, f in [("Linearly Separable", "linearly_separable.npy"),
                 ("Linearly Inseparable", "linearly_inseparable.npy")]:
        # Load dataset
        f = "../datasets/" + f
        df = load_dataset(f)
        X, y = df[0], df[1]
        # Fit Perceptron and record loss in each fit iteration
        loss = []
        perceptron = Perceptron(max_iter=1000, callback=lambda _p, _x, _y: loss.append(_p.loss(X, y)))
        perceptron.fit(X, y)

        # Plot figure of loss as function of fitting iteration
        fig.add_trace(go.Scatter(x=np.arange(len(loss)), y=loss, name=n), row=row, col=1)
        fig.update_xaxes(title_text='Iterarion', row=row)
        fig.update_yaxes(title_text='Loss', row=row)
        row += 1



    fig.show()


def get_ellipse(mu: np.ndarray, cov: np.ndarray):
    """
    Draw an ellipse centered at given location and according to specified covariance matrix

    Parameters
    ----------
    mu : ndarray of shape (2,)
        Center of ellipse

    cov: ndarray of shape (2,2)
        Covariance of Gaussian

    Returns
    -------
        scatter: A plotly trace object of the ellipse
    """
    l1, l2 = tuple(np.linalg.eigvalsh(cov)[::-1])
    theta = atan2(l1 - cov[0, 0], cov[0, 1]) if cov[0, 1] != 0 else (np.pi / 2 if cov[0, 0] < cov[1, 1] else 0)
    t = np.linspace(0, 2 * pi, 100)
    xs = (l1 * np.cos(theta) * np.cos(t)) - (l2 * np.sin(theta) * np.sin(t))
    ys = (l1 * np.sin(theta) * np.cos(t)) + (l2 * np.cos(theta) * np.sin(t))

    return go.Scatter(x=mu[0] + xs, y=mu[1] + ys, mode="lines", marker_color="black")


def compare_gaussian_classifiers():
    """
    Fit both Gaussian Naive Bayes and LDA classifiers on both gaussians1 and gaussians2 datasets
    """
    symbols = np.array(["circle", "square", "triangle-up"])
    for f in ["gaussian1.npy", "gaussian2.npy"]:
        # Load dataset
        f = "../datasets/" + f
        df = load_dataset(f)
        # Fit models and predict over training set
        X, y = df[0], df[1]
        gaussian_nb = GaussianNaiveBayes().fit(X, y)
        lda = LDA().fit(X, y)
        gaussian_nb_pred = gaussian_nb.predict(X)
        lda_pred = lda.predict(X)

        # Plot a figure with two suplots, showing the Gaussian Naive Bayes predictions on the left and LDA predictions
        # on the right. Plot title should specify dataset used and subplot titles should specify algorithm and accuracy
        # Create subplots
        from IMLearn.metrics import accuracy
        model_names = [f'LDA with accuracy {accuracy(y, lda_pred)}',
                       f'Gaussian Naive Bayes with accuracy {accuracy(y, gaussian_nb_pred)}']
        fig = make_subplots(rows=1, cols=2, subplot_titles=[rf"$\textbf{{{m}}}$" for m in model_names])
        # Add scatter plots to subplots
        fig.add_trace(go.Scatter(x=X[:,0], y=X[:,1], mode="markers",marker=dict(color=gaussian_nb_pred, symbol=symbols[y])), row=1, col=1)
        fig.add_trace(go.Scatter(x=X[:,0], y=X[:,1], mode="markers",marker=dict(color=lda_pred, symbol=symbols[y])), row=1, col=2)
        # Add ellipses to subplots
        for i in np.unique(y):
            fig.add_trace(get_ellipse(lda.mu_[i], lda.cov_), row=1, col=1)
            fig.add_trace(go.Scatter(x=[lda.mu_[i][0]], y=[lda.mu_[i][1]], mode="markers", marker=dict(color="black", symbol="x", size=10)), row=1, col=1)
            fig.add_trace(get_ellipse(gaussian_nb.mu_[i], np.diag(gaussian_nb.vars_[i])), row=1, col=2)
            fig.add_trace(go.Scatter(x=[gaussian_nb.mu_[i][0]], y=[gaussian_nb.mu_[i][1]], mode="markers", marker=dict(color="black", symbol="x", size=10)), row=1, col=2)


        fig.show()



        # Add traces for data-points setting symbols and colors
        # raise NotImplementedError()

        # Add `X` dots specifying fitted Gaussians' means
        # raise NotImplementedError()

        # Add ellipses depicting the covariances of the fitted Gaussians
        # raise NotImplementedError()


if __name__ == '__main__':
    np.random.seed(0)
    run_perceptron()
    compare_gaussian_classifiers()