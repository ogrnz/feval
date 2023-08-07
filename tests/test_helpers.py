import unittest

import numpy as np
import numpy.testing as npt

from feval import helpers


class Test(unittest.TestCase):
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

        npt.assert_array_equal(
            L[:, 0], np.abs(self.y - self.f0) / np.maximum(np.abs(self.y), epsilon)
        )
        npt.assert_array_equal(
            L[:, 1], np.abs(self.y - self.f1) / np.maximum(np.abs(self.y), epsilon)
        )

    def test_se(self):
        L = helpers.se(self.y, self.F)

        npt.assert_array_equal(L[:, 0], (self.y - self.f0) ** 2)
        npt.assert_array_equal(L[:, 1], (self.y - self.f1) ** 2)

    def test_ae_shape_mismatch(self):
        with self.assertRaises(ValueError):
            helpers.ae(np.array([1, 2]), np.array([[1, 2], [3, 4], [5, 6]]))

    def test_ape_zero_y_true(self):
        y = np.array([0, 0, 0, 0])
        F = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        L = helpers.ape(y, F)
        self.assertTrue(np.all(L[:, 0] > 1e10))  # Check if all values are big indeed

    def test_vec_to_col(self):
        vec = np.array([1, 2, 3])
        col_vec = helpers.vec_to_col(vec)
        self.assertEqual(col_vec.shape, (3, 1))
        npt.assert_array_equal(col_vec, np.array([[1], [2], [3]]))

    def test_ae_with_list(self):
        y = [0, 0, 0, 0]
        F = [[1, 2], [3, 4], [5, 6], [7, 8]]

        with self.assertRaises(TypeError):
            helpers.ae(y, F)


if __name__ == "__main__":
    unittest.main()
