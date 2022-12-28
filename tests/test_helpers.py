from unittest import TestCase

import numpy as np
import numpy.testing as npt

from feval import helpers


class Test(TestCase):
    f0 = None
    f1 = None
    F = None

    @classmethod
    def setUpClass(cls):
        cls.f0 = np.arange(4)
        cls.f1 = np.arange(4, 8)
        cls.F = np.vstack([cls.f0, cls.f1]).T
        cls.y = np.zeros(cls.F.shape[0])

    def test_ae(self):
        L = helpers.ae(self.y, self.F)

        npt.assert_array_equal(L[:, 0], np.abs(self.y - self.f0))
        npt.assert_array_equal(L[:, 1], np.abs(self.y - self.f1))

    def test_ape(self):
        L = helpers.ape(self.y, self.F)
        epsilon = np.finfo(np.float64).eps

        npt.assert_array_equal(L[:, 0], np.abs(self.y - self.f0) / np.maximum(np.abs(self.y), epsilon))
        npt.assert_array_equal(L[:, 1], np.abs(self.y - self.f1) / np.maximum(np.abs(self.y), epsilon))

    def test_se(self):
        L = helpers.se(self.y, self.F)

        npt.assert_array_equal(L[:, 0], (self.y - self.f0) ** 2)
        npt.assert_array_equal(L[:, 1], (self.y - self.f1) ** 2)
