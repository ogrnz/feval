import unittest

import arch.covariance.kernel as kernels
import numpy as np
import numpy.testing as npt

from feval import gw, mgw, cmcs, elim_rule, compute_covariance
from feval import helpers
from fixtures import data


class ComputeCovarianceTests(unittest.TestCase):
    def setUp(self):
        self.reg = np.array(
            [
                [0.42706902, 0.38860252],
                [0.41936226, 0.53305537],
                [0.05664999, 0.49064741],
                [0.05664999, 0.29064741],
                [0.86664999, 0.69064741],
            ]
        )
        self.Dbar = np.mean(self.reg, axis=0)

    def test_sample_covariance(self):
        cov_matrix = compute_covariance(self.reg, Dbar=self.Dbar, covar_style="sample")
        expected_cov = (
            (self.reg - self.Dbar).T @ (self.reg - self.Dbar) / (len(self.reg) - 1)
        )
        np.testing.assert_array_almost_equal(cov_matrix, expected_cov)

    def test_string_kernel(self):
        kernel_name = "Bartlett"
        cov_matrix = compute_covariance(
            self.reg, Dbar=self.Dbar, covar_style="hac", kernel=kernel_name, bw=2
        )

        kerfunc = getattr(kernels, kernel_name)
        ker = kerfunc(self.reg, bandwidth=2)
        expected_cov = ker.cov.long_run
        np.testing.assert_array_almost_equal(cov_matrix, expected_cov)

    def test_callable_kernel(self):
        # Simple custom callable that computes covariance
        def custom_kernel(data, **kwargs):
            return np.cov(data, rowvar=False)

        cov_matrix = compute_covariance(
            self.reg, Dbar=self.Dbar, covar_style="hac", kernel=custom_kernel
        )
        expected_cov = np.cov(self.reg, rowvar=False)
        np.testing.assert_array_almost_equal(cov_matrix, expected_cov)

    def test_unsupported_kernel_type(self):
        with self.assertRaises(NotImplementedError):
            compute_covariance(
                self.reg, Dbar=self.Dbar, covar_style="hac", kernel=12345
            )

    def test_unsupported_covar_style(self):
        with self.assertRaises(ValueError):
            compute_covariance(
                self.reg, Dbar=self.Dbar, covar_style="unsupported_style"
            )


class GWTests(unittest.TestCase):
    def setUp(self):
        self.L = np.array(
            [
                [0.42706902, 0.38860252, 0.38860252],
                [0.41936226, 0.41936226, 0.53305537],
                [0.05664999, 0.5664999, 0.49064741],
            ]
        )
        self.tau = 1
        self.H = np.array([[1, 1], [1, 1], [1, 1]])

    def test_one_step_without_kernel(self):
        S, cval, pval = gw(self.L, self.tau, self.H)
        self.assertIsInstance(S, float)
        self.assertIsInstance(cval, float)
        self.assertIsInstance(pval, float)

    def test_multistep_callable_kernel(self):
        def custom_kernel(data, **kwargs):
            return np.cov(data, rowvar=False)

        S, cval, pval = gw(self.L, self.tau, self.H, kernel=custom_kernel)
        self.assertIsInstance(S, float)
        self.assertIsInstance(cval, float)
        self.assertIsInstance(pval, float)


class MGWTests(unittest.TestCase):
    def test_mgw_uni_uncond_noreject(self):
        """
        Equivalent to DM test.
        """
        F = np.vstack([data.f1, data.f2]).T
        L_ae = helpers.ae(data.y, F)
        S, cval, pval = mgw(L_ae)

        self.assertAlmostEqual(cval, 3.841458820694124)
        self.assertAlmostEqual(S, 1.0527686753333785)
        self.assertAlmostEqual(pval, 0.3048702967179757)  # not rejecting, as expected

        # If provided H
        S, cval, pval = mgw(L_ae, H=data.H0)

        self.assertAlmostEqual(cval, 3.841458820694124)
        self.assertAlmostEqual(S, 1.0527686753333785)
        self.assertAlmostEqual(pval, 0.3048702967179757)

    def test_mgw_uni_uncond_reject(self):
        """
        Equivalent to DM test.
        """
        F = np.vstack([data.f1, data.f2 + 0.4]).T
        L_ae = helpers.ae(data.y, F)
        S, cval, pval = mgw(L_ae)

        self.assertAlmostEqual(cval, 3.841458820694124)
        self.assertAlmostEqual(S, 17.65702423591707)
        self.assertAlmostEqual(pval, 2.6453650085911384e-05)  # rejecting, as expected

        # If provided H
        S, cval, pval = mgw(L_ae, H=data.H0)

        self.assertAlmostEqual(cval, 3.841458820694124)
        self.assertAlmostEqual(S, 17.65702423591707)
        self.assertAlmostEqual(pval, 2.6453650085911384e-05)  # rejecting, as expected

    def test_mgw_uni_cond_oneh(self):
        """
        Equivalent to GW test with one instrument.
        """
        F = np.vstack([data.f1, data.f2]).T
        L_ae = helpers.ae(data.y, F)
        S, cval, pval = mgw(L_ae, H=data.H1)

        self.assertAlmostEqual(cval, 5.991464547107979)
        self.assertAlmostEqual(S, 2.9065854327777005)
        self.assertAlmostEqual(pval, 0.23379918487065443)  # not rejecting

    def test_mgw_uni_cond_twoh(self):
        """
        Equivalent to GW test with multiple instruments.
        """
        F = np.vstack([data.f1, data.f2]).T
        L_ae = helpers.ae(data.y, F)
        S, cval, pval = mgw(L_ae, H=data.H2)

        self.assertAlmostEqual(cval, 7.814727903251179)
        self.assertAlmostEqual(S, 5.750420293470262)
        self.assertAlmostEqual(pval, 0.12440471677713105)  # not rejecting

    def test_mgw_mult_uncond_noreject(self):
        """
        Equivalent to MDM test.
        """
        F = np.vstack([data.f1, data.f2, data.f3]).T
        L_ae = helpers.ae(data.y, F)
        S, cval, pval = mgw(L_ae)

        self.assertAlmostEqual(cval, 5.991464547107979)
        self.assertAlmostEqual(S, 1.096965740777901)
        self.assertAlmostEqual(pval, 0.5778257823353824)  # not rejecting, as expected

    def test_mgw_mult_uncond_reject(self):
        """
        Equivalent to MDM test.
        """
        F = np.vstack([data.f1, data.f2 + 0.4, data.f3]).T
        L_ae = helpers.ae(data.y, F)
        S, cval, pval = mgw(L_ae)

        self.assertAlmostEqual(cval, 5.991464547107979)
        self.assertAlmostEqual(S, 21.08940508706639)
        self.assertAlmostEqual(pval, 2.63326078766557e-05)  # reject

    def test_mgw_mult_cond_oneh(self):
        """
        Actual MGW test with one instrument.
        """
        F = np.vstack([data.f1, data.f2, data.f3]).T
        L_ae = helpers.ae(data.y, F)
        S, cval, pval = mgw(L_ae, H=data.H1)

        self.assertAlmostEqual(cval, 9.487729036781154)
        self.assertAlmostEqual(S, 2.9676394824211374)
        self.assertAlmostEqual(pval, 0.5632553801596231)  # not rejecting, as expected

    def test_mgw_mult_cond_twoh(self):
        """
        Actual MGW test with multiple instruments.
        """
        F = np.vstack([data.f1, data.f2, data.f3]).T
        L_ae = helpers.ae(data.y, F)
        S, cval, pval = mgw(L_ae, H=data.H2)

        self.assertAlmostEqual(cval, 12.591587243743977)
        self.assertAlmostEqual(S, 8.938482444374896)
        self.assertAlmostEqual(pval, 0.177067449276332)  # not rejecting, as expected


class MCSTests(unittest.TestCase):
    def test_mcs_uncond_noreject(self):
        """
        Traditional MCS with MGW.
        """
        F = np.vstack([data.f1, data.f2, data.f3]).T
        L_ae = helpers.ae(data.y, F)
        mcs, S, cval, pval, removed = cmcs(L_ae)

        npt.assert_array_equal(mcs, np.array([[1, 1, 1]]))
        npt.assert_array_equal(removed, np.array([[0, 0, 0]]))
        self.assertAlmostEqual(cval, 5.991464547107979)
        self.assertAlmostEqual(S, 1.096965740777901)
        self.assertAlmostEqual(pval, 0.5778257823353824)  # not rejecting, as expected

        L_ae = helpers.ae(data.y, F)
        mcs, S, cval, pval, removed = cmcs(L_ae, H=data.H0)

        npt.assert_array_equal(mcs, np.array([[1, 1, 1]]))
        npt.assert_array_equal(removed, np.array([[0, 0, 0]]))
        self.assertAlmostEqual(cval, 5.991464547107979)
        self.assertAlmostEqual(S, 1.096965740777901)
        self.assertAlmostEqual(pval, 0.5778257823353824)  # not rejecting, as expected

    def test_mcs_uncond_reject(self):
        """
        Traditional MCS with MGW.
        """
        F = np.vstack([data.f1, data.f2 + 0.4, data.f3]).T
        L_ae = helpers.ae(data.y, F)
        mcs, S, cval, pval, removed = cmcs(L_ae)

        npt.assert_array_equal(mcs, np.array([[1, 0, 1]]))
        npt.assert_array_equal(removed, np.array([[1, 0, 0]]))
        self.assertAlmostEqual(cval, 3.841458820694124)
        self.assertAlmostEqual(S, 0.16085093233763528)
        self.assertAlmostEqual(pval, 0.6883742895715335)  # not rejecting, as expected

    def test_mcs_order_011(self):
        """
        Traditional MCS with MGW, testing ordering.
        """
        F = np.vstack([data.f1 + 0.4, data.f2, data.f3]).T
        L_ae = helpers.ae(data.y, F)
        mcs, S, cval, pval, removed = cmcs(L_ae)

        npt.assert_array_equal(mcs, np.array([[0, 1, 1]]))
        npt.assert_array_equal(
            removed, np.array([[0, 0, 0]])
        )  # Could lead to unexpected behavior

    def test_mcs_order_101(self):
        F = np.vstack([data.f1, data.f2 + 0.4, data.f3]).T
        L_ae = helpers.ae(data.y, F)
        mcs, S, cval, pval, removed = cmcs(L_ae)

        npt.assert_array_equal(mcs, np.array([[1, 0, 1]]))
        npt.assert_array_equal(removed, np.array([[1, 0, 0]]))

    def test_mcs_order_110(self):
        F = np.vstack([data.f1, data.f2, data.f3 + 0.4]).T
        L_ae = helpers.ae(data.y, F)
        mcs, S, cval, pval, removed = cmcs(L_ae)

        npt.assert_array_equal(mcs, np.array([[1, 1, 0]]))
        npt.assert_array_equal(removed, np.array([[2, 0, 0]]))

    def test_mcs_order_100(self):
        F = np.vstack([data.f1, data.f2 + 0.4, data.f3 + 0.4]).T
        L_ae = helpers.ae(data.y, F)
        mcs, S, cval, pval, removed = cmcs(L_ae)

        npt.assert_array_equal(mcs, np.array([[1, 0, 0]]))
        npt.assert_array_equal(removed, np.array([[1, 2, 0]]))


if __name__ == "__main__":
    unittest.main()
