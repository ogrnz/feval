import numpy as np


def ae(y_true: np.array, F: np.array):
    """
    Compute Absolute Error loss column-wise.

    :param y_true: (T,) vector of observations
    :param F: (Txk) matrix of forecasts
    :return: (Txk) matrix of losses
    """
    y_true = vec_to_col(y_true)
    return np.abs(y_true - F)


def ape(y_true: np.array, F: np.array):
    """
    Compute Absolute Percentage Error loss column-wise.

    Note:
        Outputs a large value when `y_true == 0`.

    :param y_true: (T,) vector of observations
    :param F: (Txk) matrix of forecasts
    :return: (Txk) matrix of losses
    """
    y_true = vec_to_col(y_true)
    epsilon = np.finfo(np.float64).eps
    return np.abs(y_true - F) / np.maximum(np.abs(y_true), epsilon)


def se(y_true: np.array, F: np.array):
    """
    Compute Squared Error loss column-wise.

    :param y_true: (T,) vector of observations
    :param F: (Txk) matrix of forecasts
    :return: (Txk) matrix of losses
    """
    y_true = vec_to_col(y_true)
    return (y_true - F) ** 2


def vec_to_col(vec: np.array) -> np.array:
    """
    Change vector into column vector

    :param vec: np.array
    :return: column vector
    """
    return vec.reshape(-1, 1)
