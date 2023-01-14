# feval

Easily evaluate your forecasts with (multivariate) Diebold-Mariano or (multivariate) Giacomini-White tests of Equal
(Conditional) Predictive Ability and MCS. A procedure reducing a pool of candidates to a best final set with equal
predictive ability is also proposed.

## Intro

The multivariate Giacomini-White test (MGW) (Borup et al., 2022) is the generalization of the Giacomini-White test (
GW) (Giacomini and White, 2006), which in turn is a generalization of the famous Diebold-Mariano (DM) (Diebold and
Mariano, 1995) test. Basically, it allows to do a test of *Conditional* Predictive Ability instead of just *Equal*
Predictive Ability (like the DM does). It's an asymptotic $\mathcal{X}^2$ test of the null

```math
H_0: E[h_t \otimes \Delta L_{t + \tau}] = 0,
```

where $\tau$ is the forecast horizon and
$\Delta L_{t + \tau} = (\Delta L_{t + \tau}^{(1)}, \dots, \Delta L_{t + \tau}^{(p)})'$ a $(p \times 1)$ vector of
loss differences, that is $\Delta L_{t + \tau}^{(1)} = \Delta L_{t + \tau}^{(i)} - \Delta L_{t + \tau}^{(i + 1)}$, for
$i = 1, \dots, p$, for a $(q \times 1)$ test function $h_t$ and the Kronecker product $\otimes$.

See the references for more information.

A small interpretation note: having a *small* p-value (usually smaller than $0.05$) indicates that the null hypothesis
of equal (conditional) predictive ability is rejected: *at least one* competing method is statistically better in the
provided sample. Check the MCS implementation to see which method is actually better.

The neat thing about the MGW test is that it reduces to:

- the univariate GW test (Giacomini and White, 2006) when comparing $k = 2$ methods with potential conditiong;
- the multivariate DM test (Mariano and Preve, 2012) when comparing $k > 2$ methods without conditioning;
- the univariate DM test (Diebold and Mariano, 1995) when comparing $k = 2$ methods without conditioning.

Hence, it can be used in those 3 cases.

However, the tests give no indication as to *which* method being compared is the best. That is done by the MCS, adapted
from Hansen (2011). See the examples below.

## Installation

```bash
# Create virtual environment
> python -m venv venv
# Activate it
> source ./venv/bin/activate
# Pip install from git
> pip install git+https://github.com/ogrnz/feval

# If already installed, you can also upgrade it 
> pip install --upgrade git+https://github.com/ogrnz/feval
```

## Examples

Those examples are here to illustrate how to use the package.

Unconditional MGW example:

The forecasts are equally good, on average, and hence have the same predictive ability.

```python
import numpy as np
from feval import helpers  # to easily compute losses
from feval import mgw

T = 100
F = np.vstack([np.random.rand(T), np.random.rand(T), np.random.rand(T)]).T  # random uniform forecasts [0,1)
y = np.zeros(T) + 0.5  # Target
L = helpers.ae(y, F)  # Absolute loss

S, cval, pval = mgw(L)  # Perform the test with default values

# We should get a large p-value, since both forecasts are equally bad at predicting y
print(pval)  # 0.61 (exact value can change due to randomness)
# As expected, the null of equal predictive ability is not rejected, 
# we cannot say that a model is better than another
```

Unconditional CMCS example:
The best forecast is the 3rd one, it should be the last one in the best set.

```python
import numpy as np
from feval import helpers  # to easily compute losses
from feval import cmcs

T = 100
F = np.vstack(
    [np.random.rand(T) + 0.5,
     np.random.rand(T) + 1.0,
     np.random.rand(T),  # Only this forecast is 'good'
     np.random.rand(T) - 0.3]).T
y = np.zeros(T) + 0.5  # Target
L = helpers.se(y, F)  # Squared loss

# Perform the cmcs with an HAC estimator, the Bartlett kernel and a significance level of 0.01
mcs, S, cval, pval, removed = cmcs(L, alpha=0.01, covar_style="hac", kernel="Bartlett")

print(mcs)  # [0, 0, 1, 0], only the 3rd model is included in the best set
```

Conditional MCS:
1st and 3rd forecasts are equally good, while the others are biased. Here, the use of instruments is useless, but serves
as an illustration.

```python
import numpy as np
from feval import helpers  # to easily compute losses
from feval import cmcs

# Conditional MCS
T = 101  # Set 1 more to allow 1 lag computation as instrument
F = np.vstack(
    [np.random.rand(T),  # This forecast is 'good'
     np.random.rand(T) + 0.5,
     np.random.rand(T),  # This forecast is 'good'
     np.random.rand(T) - 0.5]).T
y = np.zeros(T) + 0.5  # Target
L = helpers.se(y, F)  # Squared loss

# Compute instruments as lags of loss differences
# Instruments useless here, but to illustrate its use
D = np.diff(L, axis=1)
D = np.roll(D, 1, axis=0)[:-1]
H = np.vstack([np.ones(T - 1), D.T]).T  # Instruments, a constant + lags of loss differences

# Perform the cmcs with an HAC estimator with Parzen kernel
mcs, S, cval, pval, removed = cmcs(L[:-1, :], H=H, covar_style="hac", kernel="Parzen")

print(mcs)  # [1, 0, 1, 0], only the 1st and 3rd models are included in the best set
```

## Development

```bash
# Create virtual environment
> python -m venv venv
# Activate it
> source ./venv/bin/activate
# Clone the repo
> git clone https://github.com/ogrnz/feval
# Install it in editable mode for your user
> pip install -U --editable feval 
```

Don't forget to test your code with the scripts in `./tests`!

### Notes

- The statistical tests have been "translated" from their matlab code and unified under a single API;
- To keep minimal requirements, the tests handle numpy arrays only
- HAC covariance estimators are computed by
  the [Arch package](https://arch.readthedocs.io/en/latest/covariance/covariance.html), but can be computed by any
  Callable.

## References

- Borup, Daniel and Eriksen, Jonas Nygaard and Kjær, Mads Markvart and Thyrsgaard, Martin,
  [Predicting Bond Return Predictability](http://dx.doi.org/10.2139/ssrn.3513340).
- Diebold, F.X., and R.S. Mariano (1995)
  ‘[Comparing Predictive Accuracy](https://www.tandfonline.com/doi/abs/10.1198/073500102753410444?journalCode=ubes20),’
  Journal
  of Business and Economic Statistics 13, 253–263.
- Giacomini, R., & White, H. (2006)
  . [Tests of conditional predictive ability](https://onlinelibrary.wiley.com/doi/pdf/10.1111/j.1468-0262.2006.00718.x?casa_token=v5yp0mfNHWsAAAAA:_QfioyU_tyBuN-lU_IXcyb3yizOxA7KSMhhA94wNwokFJj5jAHgnsgXVsClT3_5MdqMK0NJPEt4TxBnN)
  .
  Econometrica, 74(6), 1545-1578.
- Hansen, P. R., Lunde, A., & Nason, J. M. (2011)
  . [The model confidence set](https://onlinelibrary.wiley.com/doi/pdf/10.3982/ECTA5771?casa_token=W_wNjvfGBEkAAAAA:EGQ4b2xpaI-S_6ALXL8F60Pg2FR42wxa4IpJ0p2RAIhAl26elh3K40qI7Xki4F7Zlyr1SLam5ag0iPdb)
  . Econometrica, 79(2), 453-497.
- Mariano, R. S., & Preve, D. (2012)
  . [Statistical tests for multiple forecast comparison](https://ink.library.smu.edu.sg/cgi/viewcontent.cgi?article=3330&context=soe_research)
  .
  Journal of econometrics, 169(1), 123-130.

