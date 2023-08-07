from typing import Callable, Literal, Optional, Union

import numpy as np

import arch.covariance.kernel as kernels
from scipy.stats import chi2


def gw(
    L: np.array,
    tau: int,
    H: Optional[np.array] = None,
    covar_style: str = "sample",
    kernel: Optional[Union[str, Callable]] = None,
    bw: Optional[int] = None,
    kernel_kwargs: Optional[dict] = None,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """
    Test of Equal Conditional Predictive Ability by Giacomini and White (2006).
    Used here for testing and debugging but made available through the package interface.

    This is a reimplementation from the MATLAB code provided by
    Giacomini (https://gist.github.com/ogrnz/91f37140011d1c2447934766274c4070)

    References:
        - Giacomini, R., & White, H. (2006). Tests of conditional predictive ability. Econometrica, 74(6), 1545-1578.

    :param L: (Tx2) array of forecast losses
    :param H: (Txq) array of instruments. If `None` provided, defaults to the unconditional EPA (DM test)
    :param tau: Forecast horizon
    :param covar_style: (default 'sample')
        How to compute the covariance matrix.
        Either the sample covariance ('sample') or an HAC estimator ('hac').
    :param kernel: (default `None`)
        If multistep forecast (`tau` > 1), the covariance matrix needs to take
        into account the correlation structure (`HAC` estimator).
        Original implementation uses a Bartlett kernel with bandwidth `tau - 1`.
        If a `str`, must match one of `arch` package variance estimator:
         > https://arch.readthedocs.io/en/latest/covariance/covariance.html
        If a `Callable`, must simply return a (qxq) covariance matrix (see arg `H`).
    :param bw: (default `None`)
        Bandwidth of the `kernel`. Typically set to `forecasting horizon - 1`.
        If set to `None`, will let the kernel compute the optimal bandwidth if supported.
    :param kernel_kwargs: (default `None`)
        An optional dict of `argument: value` passed to `kernel`.
        If `kernel` is a `Callable`, the eventual bandwidth must be passed here.
    :param alpha: Significance level
    :return: tuple(S, crit_val, p_val)
        S: test statistic,
        cval: critical value for significance lvl,
        pval: p-value of test
    """
    if kernel_kwargs is None:
        kernel_kwargs = {}

    T, q = L.shape[0], H.shape[1] if H is not None else 1
    d = L[:, 0] - L[:, 1]

    # Default instruments for unconditional EPA
    H = np.ones((T, 1)) if H is None else H

    reg = H * d[:, np.newaxis]

    if tau == 1:  # One-step
        beta = np.linalg.lstsq(reg, np.ones(T), rcond=None)[0][0]
        residuals = np.ones((T, 1)) - (beta * reg)
        mean_residuals = np.mean(residuals, axis=1)
        S = T * (1 - np.mean(mean_residuals**2))
    else:  # Multistep
        omega = compute_covariance(reg, covar_style, kernel, bw, kernel_kwargs)
        zbar = reg.mean().T
        S = T * zbar.T @ np.linalg.pinv(omega) @ zbar

    dof = reg.shape[1]
    cval = chi2.ppf(1 - alpha, dof)
    pval = 1 - chi2.cdf(abs(S), dof)

    return S, cval, pval


def mgw(
    L: np.array,
    H: Optional[np.array] = None,
    covar_style: Literal["sample", "hac"] = "sample",
    kernel: Optional[Union[str, Callable]] = None,
    bw: Optional[int] = None,
    kernel_kwargs: Optional[dict] = None,
    alpha: float = 0.05,
):
    """
    Implements the multivariate Giacomini-White (MGW) (Borup et al., 2022) test of equal predictive ability.

    This is a reimplementation from the MATLAB code provided by
    Borup (https://sites.google.com/view/danielborup/research)

    Notes:
        If only 2 models are compared, it reduces to the Giacomini-White test (GW) (Giacomini and White, 2006)
        If further no conditioning information H is given, it reduces to the
        original Diebold-Mariano test (DM) (Diebold and Mariano, 1995)
        If more than 2 models are compared but with no conditioning information H,
        it reduces to multivariate Diebold-Mariano (MDM) (Mariano and Preve, 2012)

    References:
        - Borup, Daniel and Eriksen, Jonas Nygaard and Kjær, Mads Markvart and Thyrsgaard, Martin,
        Predicting Bond Return Predictability. Available at http://dx.doi.org/10.2139/ssrn.3513340
        - Diebold, F.X., and R.S. Mariano (1995) ‘Comparing Predictive Accuracy,’ Journal
        of Business and Economic Statistics 13, 253–263.
        - Giacomini, R., & White, H. (2006). Tests of conditional predictive ability.
        Econometrica, 74(6), 1545-1578.
        - Mariano, R. S., & Preve, D. (2012). Statistical tests for multiple forecast comparison.
        Journal of econometrics, 169(1), 123-130.

    :param L:
        Txk matrix of losses of k models with T forecasts.
    :param H: (default `None`)
        Txq matrix of a constant and information set (test function).
        If not provided, set to a (Tx1) column vector of 1, amounts to the
        unconditional MWG test, which is equivalent to the multivariate Diebold-Mariano (Mariano and Preve, 2012).
    :param covar_style: (default 'sample')
        How to compute the covariance matrix.
        Either the sample covariance ('sample') or an HAC estimator ('hac').
    :param kernel: (default `None`)
        If covariance matrix is an HAC estimator, what type to compute.
        If a `str`, must match one of `arch` package variance estimator:
         > https://arch.readthedocs.io/en/latest/covariance/covariance.html
        If a `Callable`, must simply return a
    :param bw: (default `None`)
        Bandwidth of the `kernel`. Typically set to `forecasting horizon - 1`.
        If set to `None`, will let the kernel compute the optimal bandwidth if supported.
    :param kernel_kwargs: (default `None`)
        An optional dict of `argument: value` passed to `kernel`.
        If `kernel` is a `Callable`, the eventual bandwidth must be passed here.
    :param alpha: (default 0.05)
        Significance level.
    :return: tuple(S, cval, pval)
        S: float, the computed test statistic
        cval: float, the corresponding critical value
        pval: float, the p-value of S.
    """
    validate_args(L, covar_style, kernel, bw)

    if kernel_kwargs is None:
        kernel_kwargs = {}

    T, p = L.shape[0], L.shape[1] - 1
    H = np.ones((T, 1)) if H is None else H  # default to unconditional EPA
    D = np.diff(L, axis=1)
    reg = np.array([np.kron(h, d) for h, d in zip(H, D)])

    Dbar = np.mean(reg, axis=0)
    omega = compute_covariance(reg, Dbar, covar_style, kernel, bw, kernel_kwargs)

    dof = H.shape[1] * p
    S = (T * Dbar @ np.linalg.pinv(omega) @ Dbar.T).item()
    cval = chi2.ppf(1 - alpha, dof)
    pval = 1 - chi2.cdf(S, dof)

    return S, cval, pval


def cmcs(L: np.array, H: Optional[np.array] = None, alpha: float = 0.05, **kwargs):
    """
    Perform the Conditional Model Confidence Set (CMCS).
    The MCS procedure from Hansen (2011) is adapted to use MGW (Borup et al., 2022)
    instead of bootstrapping the critical values. Allows to reduce an initial set of models to a
    set of models with equal (conditional) predictive ability.
    Also, allows to use conditioning information (`H`, hence the 'Conditional'),
    to get the best MCS based on expected future loss.

    This is a reimplementation from the MATLAB code provided by
    Borup (https://sites.google.com/view/danielborup/research)

    References:
        - Borup, Daniel and Eriksen, Jonas Nygaard and Kjær, Mads Markvart and Thyrsgaard, Martin,
        Predicting Bond Return Predictability. Available at http://dx.doi.org/10.2139/ssrn.3513340
        - Hansen, P. R., Lunde, A., & Nason, J. M. (2011). The model confidence set. Econometrica, 79(2), 453-497.

    :param L:
        (Txk) matrix of losses of k models with T forecasts.
    :param H: (default `None`)
        (Txq) matrix of a constant and information set (test function).
        If not provided, set to a (Tx1) column vector of 1, amounts to the
        unconditional MWG test, which is equivalent to the multivariate Diebold-Mariano (Mariano and Preve, 2012).
    :param alpha: (default 0.05)
        Significance level used in the MGW test.
    :param **kwargs: Arguments passed to `feval.mgw`. Usually define covariance estimator and such.
    :return: tuple(mcs, S, cval, pval, removed)
        mcs: (1xk) np.array where models included in the best model confidence set are noted as 1.
        S: float, the computed test statistic of the last test.
        cval: float, the corresponding critical value.
        pval: float, the p-value of S.
        removed: (1xk) np.array where a column represents an algorithm cycle.
            That way, we can see which model index was removed at which iteration.
    """
    # Initialize
    T = L.shape[0]
    k = L.shape[1]
    if H is None:
        H = np.ones((T, 1))

    # Init loop
    S, cval, pval = np.inf, 1, 1
    mcs = np.ones((1, k))
    removed = np.zeros((1, k))

    j = 0
    while S > cval:
        # Create L_to_use, the losses of models still in MCS
        L_to_use = L[:, (mcs == 1)[0]]

        if L_to_use.shape[1] == 1:  # Only 1 model left in set
            break

        # Perform MGW
        S, cval, pval = mgw(L_to_use, H, alpha=alpha, **kwargs)

        # H0 still rejected, apply elimination criterion
        if S > cval:
            mcs, removed[0, j] = elim_rule(L, mcs, H)

        j += 1

    return mcs, S, cval, pval, removed


def elim_rule(L: np.array, mcs: np.array, H: Optional[np.array] = None):
    """
    Elimination rule that allows to rank losses based on expected future loss given the information set `H`.
    If `H` is a vector of constant, it amounts to ranking losses based on average loss.
    See Borup et al. (2022) and Hansen (2011).

    This is a reimplementation from the MATLAB code provided by
    Borup (https://sites.google.com/view/danielborup/research)

    References:
        - Borup, Daniel and Eriksen, Jonas Nygaard and Kjær, Mads Markvart and Thyrsgaard, Martin,
        Predicting Bond Return Predictability. Available at http://dx.doi.org/10.2139/ssrn.3513340
        - Hansen, P. R., Lunde, A., & Nason, J. M. (2011). The model confidence set. Econometrica, 79(2), 453-497.

    :param L:
        (Txk) matrix of losses of k models with T forecasts.
    :param mcs:
        (1xk) vector of current model confidence set, where the least performing model will be eliminated.
    :param H: (default `None`)
        (Txq) matrix of a constant and information set (test function).

    :return: tuple(mcs, removed)
        mcs: (1xk) np.array where models included in the best model confidence set are noted as 1.
        removed: (1xk) np.array where a column represents an algorithm cycle.
            That way, we can see which model index was removed at which iteration.
    """
    # Initialize
    k = mcs.shape[1]
    q = H.shape[1]
    new_k = np.count_nonzero(mcs)

    if L.shape[1] != k:
        raise ValueError(f"Dimensions of {L.shape[1]=} do not match {mcs.shape[1]=}.")

    L_to_use = np.zeros((L.shape[0], new_k))
    curr_set = np.zeros((1, new_k))
    j = 0
    for i in range(k):  # TODO could be simplified?
        if mcs[0, i] == 1:
            L_to_use[:, j] = L[:, i]
            curr_set[0, j] = i
            j += 1

    combinations = np.arange(0, j).reshape(1, -1)  # TODO why matrix? could be vect
    L_hat = np.zeros(combinations.shape)

    # Estimate
    for j in range(combinations.shape[0]):  # TODO no loop if vect
        L_intra_use = L_to_use[:, combinations[j, :]]  # TODO directly call j?

        deltas = np.zeros((q, new_k - 1))
        for i in range(L_to_use.shape[1] - 1):
            Y_used = L_intra_use[:, i + 1] - L_intra_use[:, i]
            Y_used = Y_used.reshape(-1, 1)
            deltas[:, i] = (np.linalg.inv(H.T @ H) @ H.T @ Y_used).reshape(
                -1,
            )

        delta_L_hat = (deltas.T @ H[-1, :].T).reshape(-1, 1)
        starting_point = combinations[combinations == 0]  # should always return 1 idx

        # Normalize
        vL_hat = np.zeros(L_hat.shape)
        vL_hat[0, 0] = 1
        for i in range(L_to_use.shape[1] - 1):
            vL_hat[0, i + 1] = vL_hat[0, i] + delta_L_hat[i, 0]
        L_hat[j, :] = np.divide(vL_hat, vL_hat[0, starting_point].item())

    # Rank losses
    indx = np.argmax(L_hat)
    col = np.unique(combinations[0, indx])

    # Update mcs
    mcs[0, curr_set[0, col].astype(int)] = 0
    removed = curr_set[0, col]
    return mcs, removed


def validate_args(L, covar_style, kernel, bw):
    if kernel and covar_style == "sample":
        raise ValueError(f"{kernel=} incompatible with {covar_style=}.")
    if not kernel and covar_style == "hac":
        raise ValueError("Set `kernel` when using an HAC estimator.")
    if bw and covar_style == "sample":
        raise ValueError(f"{bw=} incompatible with {covar_style=}.")
    if L.shape[1] < 2:
        raise ValueError(f"Not enough columns for matrix of losses {L.shape[1]=}.")


def compute_covariance(
    reg: np.array,
    Dbar: np.array,
    covar_style: str,
    kernel: Optional[Union[str, Callable]] = None,
    bw: Optional[int] = None,
    kernel_kwargs: Optional[dict] = None,
) -> np.array:
    """
    Compute the covariance matrix omega for the given regression residuals and kernel.

    :param reg: Residuals from the regression.
    :param Dbar: Mean of the regression residuals.
    :param covar_style: How to compute the covariance matrix. Either 'sample' or 'hac'.
    :param kernel:
        The kernel function or name.
        If it's a string, it should match one of the `arch` package variance estimator.
        If it's a callable, it should return a covariance matrix.
    :param bw: Bandwidth for the kernel. If None, the kernel might compute the optimal bandwidth.
    :param kernel_kwargs: Additional keyword arguments to be passed to the kernel function.
    :return np.array: The computed covariance matrix omega.
    """
    if kernel_kwargs is None:
        kernel_kwargs = {}

    if covar_style == "sample":
        return (reg - Dbar).T @ (reg - Dbar) / (len(reg) - 1)
    elif covar_style == "hac":
        if callable(kernel):
            return kernel(reg, **kernel_kwargs)
        elif isinstance(kernel, str) and hasattr(kernels, kernel):  # Arch covariance
            kerfunc = getattr(kernels, kernel)
            ker = kerfunc(reg, bandwidth=bw, **kernel_kwargs)
            return ker.cov.long_run
        else:
            raise NotImplementedError("Kernel not recognized or not implemented")
    else:
        raise ValueError(f"Unsupported covariance style: {covar_style}")
