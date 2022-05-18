import numpy as np
from typing import Tuple
from IMLearn.metalearners.adaboost import AdaBoost
from IMLearn.learners.classifiers import DecisionStump
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def generate_data(n: int, noise_ratio: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a dataset in R^2 of specified size

    Parameters
    ----------
    n: int
        Number of samples to generate

    noise_ratio: float
        Ratio of labels to invert

    Returns
    -------
    X: np.ndarray of shape (n_samples,2)
        Design matrix of samples

    y: np.ndarray of shape (n_samples,)
        Labels of samples
    """
    '''
    generate samples X with shape: (num_samples, 2) and labels y with shape (num_samples).
    num_samples: the number of samples to generate
    noise_ratio: invert the label for this ratio of the samples
    '''
    X, y = np.random.rand(n, 2) * 2 - 1, np.ones(n)
    y[np.sum(X ** 2, axis=1) < 0.5 ** 2] = -1
    y[np.random.choice(n, int(noise_ratio * n))] *= -1
    return X, y


def fit_and_evaluate_adaboost(noise, n_learners=250, train_size=5000, test_size=500):
    (train_X, train_y), (test_X, test_y) = generate_data(train_size, noise), generate_data(test_size, noise)

    # Question 1: Train- and test errors of AdaBoost in noiseless case
    adaboost = AdaBoost(DecisionStump, n_learners)
    adaboost.fit(train_X, train_y)
    test_errors = []
    train_errors = []
    for i in range(n_learners):
        train_errors.append(adaboost.partial_loss(train_X, train_y, i))
        test_errors.append(adaboost.partial_loss(test_X, test_y, i))
    go.Scatter(x=np.arange(n_learners), y=train_errors, name='Train Errors', mode='lines')
    go.Scatter(x=np.arange(n_learners), y=test_errors, name='Test Errors', mode='lines')
    go.Layout(title='Train and Test Errors of AdaBoost')
    fig = go.Figure(data=[go.Scatter(x=np.arange(n_learners), y=train_errors, name='Train Errors', mode='lines'),
                          go.Scatter(x=np.arange(n_learners), y=test_errors, name='Test Errors', mode='lines')])
    fig.show()

    # Question 2: Plotting decision surfaces
    Ts = [5, 50, 100, 250]
    # Ts = [5, 10, 15, 20]
    lims = np.array([np.r_[train_X, test_X].min(axis=0), np.r_[train_X, test_X].max(axis=0)]).T + np.array([-.1, .1])
    print(lims)
    fig = make_subplots(rows=2, cols=2, subplot_titles=[f"Desicion surface for T= {t}" for t in Ts],
                        horizontal_spacing=0.01, vertical_spacing=.03)

    X = test_X
    y = test_y
    symbols = np.array(["circle", "x"])

    def predict_wrapper(T):
        return lambda X: adaboost.partial_predict(X, T)

    for i, t in enumerate(Ts):
        partial_predict_t = predict_wrapper(t)
        pred_y = adaboost.partial_predict(X, t)
        fig.add_traces([decision_surface(partial_predict_t, lims[0], lims[1], showscale=False),
                        go.Scatter(x=X[:, 0], y=X[:, 1], mode="markers", showlegend=False,
                                   marker=dict(color=y.astype(int), symbol=symbols[((pred_y + 1) / 2).astype(int)],
                                               colorscale=[custom[0], custom[-1]],
                                               line=dict(color="black", width=1)))],
                       rows=(i // 2) + 1, cols=(i % 2) + 1)
    fig.update_layout(title_text='Decision surfaces for different T', width=800, height=800)
    fig.show()

    # Question 3: Decision surface of best performing ensemble
    best_T = np.argmin(test_errors)
    print(f"Best T: {best_T}")
    partial_predict_t = predict_wrapper(best_T)
    pred_y = adaboost.partial_predict(X, best_T)
    fig = go.Figure(data=[decision_surface(partial_predict_t, lims[0], lims[1], showscale=False),
                          go.Scatter(x=X[:, 0], y=X[:, 1], mode="markers", showlegend=False,
                                     marker=dict(color=y.astype(int), symbol=symbols[((pred_y + 1) / 2).astype(int)],
                                                 colorscale=[custom[0], custom[-1]],
                                                 line=dict(color="black", width=1)))])
    fig.update_layout(title_text=f"Decision surface for best T= {best_T}", width=800, height=800)
    fig.show()
    # Question 4: Decision surface with weighted samples
    pred_y = adaboost.predict(train_X)
    go.Figure(data=[decision_surface(adaboost.predict, lims[0], lims[1], showscale=False),
                    go.Scatter(x=train_X[:, 0], y=train_X[:, 1], mode="markers", showlegend=False,
                               marker=dict(color=train_y.astype(int), symbol=symbols[((pred_y + 1) / 2).astype(int)],
                                           colorscale=[custom[0], custom[-1]],
                                           line=dict(color="black", width=1),
                                           size=(adaboost.D_ / (np.max(adaboost.D_)) * 5)))])
    fig.update_layout(title_text=f"Decision surface with weights", width=800, height=800)
    fig.show()

if __name__ == '__main__':
    fit_and_evaluate_adaboost(0)
    fit_and_evaluate_adaboost(0.4)
