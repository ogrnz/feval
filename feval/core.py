from typing import Optional

import numpy as np

from scipy.stats import chi2


def gw(L: np.array, tau: int, H: Optional[np.array] = None, alpha: float = 0.05, return_omega: bool = False) -> tuple:
    """
    :param L: Tx2 array of forecast losses
    :param H: Txk array of instruments. If `None` provided, defaults to the unconditional EPA
    :param tau: Forecast horizon
    :param alpha: Significance level
    :return: tuple(S, crit_val, p_val)
        S: test statistic,
        cval: critical value for significance lvl,
        pval: p-value of test
    """
    t = L.shape[0]  # Number of observations
    d = L[:, 0] - L[:, 1]  # Loss differential

    if H is None:  # Instruments (defaults to unconditional EPA)
        H = np.ones((t, 1))

    reg = np.empty(H.shape)
    for jj in range(H.shape[1]):
        reg[:, jj] = H[:, jj] * d

    omega = None
    if tau == 1:
        beta = np.linalg.lstsq(reg, np.ones(t), rcond=None)[0][0].item()
        res = np.ones(t) - beta * reg
        r2 = 1 - np.mean(res ** 2)
        S = t * r2
    else:
        zbar = reg.mean().T
        nlags = tau - 1
        omega = Bartlett(reg, bandwidth=nlags, force_int=True).cov.long_run
        S = (t * zbar.T * np.linalg.pinv(omega) * zbar).item()

    dof = reg.shape[1]
    cval = chi2.ppf(1 - alpha, dof)
    pval = 1 - chi2.cdf(abs(S), dof)

    return (S, cval, pval, omega) if return_omega else (S, cval, pval)

