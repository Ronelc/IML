from IMLearn.learners.classifiers import Perceptron, LDA, GaussianNaiveBayes
from typing import Tuple
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from math import atan2, pi
import matplotlib.pyplot as plt
import plotly.express as px


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
    for n, f in [("Linearly Separable", "linearly_separable.npy"),
                 ("Linearly Inseparable", "linearly_inseparable.npy")]:
        # Load dataset
        X, y = load_dataset("../datasets/" + f)

        # Fit Perceptron and record loss in each fit iteration
        losses = []

        # define callback function
        def callback(fit: Perceptron, x: np.ndarray, y: int):
            fit.fitted_ = True
            fit.losses = losses
            losses.append(fit.loss(x, y))

        p = Perceptron(callback=callback).fit(X, y)

        # Plot figure of loss as function of fitting iteration
        iterations = [m for m in range(0, len(losses))]
        plt.plot(iterations, losses, label="Loss per iteration")
        plt.xlabel("iteration")
        plt.ylabel("Loss")
        plt.show()


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
    theta = atan2(l1 - cov[0, 0], cov[0, 1]) if cov[0, 1] != 0 else (
        np.pi / 2 if cov[0, 0] < cov[1, 1] else 0)
    t = np.linspace(0, 2 * pi, 100)
    xs = (l1 * np.cos(theta) * np.cos(t)) - (l2 * np.sin(theta) * np.sin(t))
    ys = (l1 * np.sin(theta) * np.cos(t)) + (l2 * np.cos(theta) * np.sin(t))

    return go.Scatter(x=mu[0] + xs, y=mu[1] + ys, mode="lines",
                      marker_color="black")


def compare_gaussian_classifiers():
    """
    Fit both Gaussian Naive Bayes and LDA classifiers on both gaussians1 and gaussians2 datasets
    """
    for f in ["gaussian1.npy", "gaussian2.npy"]:
        # Load dataset
        X, y = load_dataset("../datasets/" + f)

        # Fit models and predict over training set
        lda = LDA().fit(X, y)
        gnb = GaussianNaiveBayes().fit(X, y)
        lda_predict = lda.predict(X)
        gnb_predict = gnb.predict(X)

        # Plot a figure with two subplots, showing the Gaussian Naive Bayes
        # predictions on the left and LDA predictions on the right.
        # Plot title should specify dataset used and subplot titles should
        # specify algorithm and accuracy Create subplots
        from IMLearn.metrics import accuracy
        lda_accuracy = accuracy(y, lda_predict)
        gnb_accuracy = accuracy(y, gnb_predict)

        symbols = np.array(["circle", "square"])
        accuracy_lst = ["GNB, accuracy = " + "{:.3f}".format(gnb_accuracy),
                        "LDA, accuracy = " + str(lda_accuracy)]
        fig = make_subplots(rows=1, cols=2,
                            subplot_titles=[rf"{m}" for m in accuracy_lst],
                            horizontal_spacing=0.01, vertical_spacing=.03)

        # Add traces for data-points setting symbols and colors
        for i, m in enumerate((gnb_predict, lda_predict)):
            is_eq = [1 if m[j] == y[j] else 0 for j in range(len(m))]
            fig.add_traces([go.Scatter(x=X[:, 0], y=X[:, 1], mode="markers",
                                       marker=dict(color=m,
                                                   symbol=symbols[is_eq],
                                                   line=dict(color='black')))],
                           rows=(i // 2) + 1, cols=(i % 2) + 1)

        # Add `X` dots specifying fitted Gaussians' means
        for i, mu in enumerate((gnb.mu_, lda.mu_)):
            fig.add_traces(
                [go.Scatter(x=mu[:, 0], y=mu[:, 1], mode="markers",
                            marker=dict(color='black',
                                        symbol='x',
                                        line=dict(color='black', width=2),
                                        size=15))],
                rows=(i // 2) + 1, cols=(i % 2) + 1)

        # Add ellipses depicting the covariances of the fitted Gaussians
        for k in range(len(lda.classes_)):
            fig.add_traces([get_ellipse(gnb.mu_[k], np.diag(gnb.vars_[k]))],
                           rows=1, cols=1)
        for k in range(len(lda.classes_)):
            fig.add_traces([get_ellipse(lda.mu_[k], lda.cov_)],
                           rows=1, cols=2)

        # add plot title
        fig.update_layout(
            title=rf"LDA VS GNB, dataset: {f}", margin=dict(t=100)) \
            .update_xaxes(visible=False).update_yaxes(visible=False)
        fig.write_html('tmp.html', auto_open=True)


if __name__ == '__main__':
    np.random.seed(0)
    run_perceptron()
    compare_gaussian_classifiers()
