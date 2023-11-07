from __future__ import annotations

import numpy as np


def ae(y_true: np.array, F: np.array) -> np.array:
    """
    Compute Absolute Error loss column-wise.

    :param y_true: (T,) vector of observations
    :param F: (Txk) matrix of forecasts
    :return: (Txk) matrix of losses
    """
    y_true = vec_to_col(y_true)
    check_vec_dim_compatibility(y_true, F)
    return np.abs(y_true - F)


def ape(y_true: np.array, F: np.array) -> np.array:
    """
    Compute Absolute Percentage Error loss column-wise.

    Note:
        Outputs a large value when `y_true == 0`.

    :param y_true: (T,) vector of observations
    :param F: (Txk) matrix of forecasts
    :return: (Txk) matrix of losses
    """
    y_true = vec_to_col(y_true)
    check_vec_dim_compatibility(y_true, F)
    epsilon = np.finfo(np.float64).eps
    return np.abs(y_true - F) / np.maximum(np.abs(y_true), epsilon)


def se(y_true: np.array, F: np.array) -> np.array:
    """
    Compute Squared Error loss column-wise.

    :param y_true: (T,) vector of observations
    :param F: (Txk) matrix of forecasts
    :return: (Txk) matrix of losses
    """
    y_true = vec_to_col(y_true)
    check_vec_dim_compatibility(y_true, F)
    return (y_true - F) ** 2


def vec_to_col(vec: np.array) -> np.array:
    """
    Convert vector into column vector.

    :param vec: np.array
    :return: column vector
    """
    if not isinstance(vec, np.ndarray):
        raise TypeError(f"vec must be an np.ndarray, currently ({type(vec)=})")

    if vec.shape == (-1, 1):
        return
    return vec.reshape(-1, 1)


def check_vec_dim_compatibility(v1: np.array, v2: np.array) -> None:
    """
    Check if 2 vectors or matrices are compatible for substraction based on their shapes.

    :param v1: (Txk) vector or matrix
    :param v2: (Txk) vector or matrix
    :raises ValueError: If v1 and v2 are not compatible
    """
    if v1.shape[0] != v2.shape[0]:
        raise ValueError(
            f"The length of v1 ({v1.shape[0]}) does not match the number of rows in v2 ({v2.shape[0]})."
        )
