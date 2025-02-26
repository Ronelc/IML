import numpy as np


def mean_square_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate MSE loss

    Parameters
    ----------
    y_true: ndarray of shape (n_samples, )
        True response values
    y_pred: ndarray of shape (n_samples, )
        Predicted response values

    Returns
    -------
    MSE of given predictions
    """
    return np.nanmean(((y_true - y_pred) ** 2))


def misclassification_error(y_true: np.ndarray, y_pred: np.ndarray,
                            normalize: bool = True) -> float:
    """
    Calculate misclassification loss

    Parameters
    ----------
    y_true: ndarray of shape (n_samples, )
        True response values
    y_pred: ndarray of shape (n_samples, )
        Predicted response values
    normalize: bool, default = True
        Normalize by number of samples or not

    Returns
    -------
    Misclassification of given predictions
    """
    sum = 0
    if normalize:
        for i in range(len(y_true)):
            if y_true[i] != y_pred[i]:
                sum += 1
        return sum / len(y_true)
    return sum


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate accuracy of given predictions

    Parameters
    ----------
    y_true: ndarray of shape (n_samples, )
        True response values
    y_pred: ndarray of shape (n_samples, )
        Predicted response values

    Returns
    -------
    Accuracy of given predictions
    """
    # calculate thr formula (TP + TN)/(P + N)
    TP, TN, P, N = 0, 0, 0, 0
    for i in range(len(y_true)):
        if y_pred[i] == 1:
            P += 1
        else:
            N += 1
        #     todo check if get in to the if
        if y_true[i] == y_pred[i]:
            if y_true[i] == 1:
                TP += 1
            else:
                TN += 1
    return (TP + TN) / (P + N)


def cross_entropy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate the cross entropy of given predictions

    Parameters
    ----------
    y_true: ndarray of shape (n_samples, )
        True response values
    y_pred: ndarray of shape (n_samples, )
        Predicted response values

    Returns
    -------
    Cross entropy of given predictions
    """
    raise NotImplementedError()
