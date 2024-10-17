import unittest
import pandas as pd
import os
import numpy as np
from pandas.testing import assert_frame_equal
from numpy.testing import assert_array_equal
from ..utils import readmatrix, linspace, meshgrid, integral, integral2, randperm, randi, rand, normrnd, chi2rnd
from ..utils import rmmissing, nanmean, fillmissing_with_mean

class TestReadMatrix(unittest.TestCase):

    def setUp(self):
        """
        Create a temporary Excel file for testing.
        """
        self.test_file = 'test_data.xlsx'

        # Create a DataFrame to write to Excel
        self.data = pd.DataFrame({
            'A': [1, 2, 3, 4, 5],
            'B': [6, 7, 8, 9, 10],
            'C': [11, 12, 13, 14, 15],
            'D': [16, 17, 18, 19, 20],
            'E': [21, 22, 23, 24, 25]
        })

        # Save the DataFrame to an Excel file
        self.data.to_excel(self.test_file, index=False, header=False)

    def test_read_specific_range(self):
        """
        Test reading a specific range (A1:E5) from the Excel file.
        """
        expected_output = pd.DataFrame({
            0: [1, 2, 3, 4, 5],
            1: [6, 7, 8, 9, 10],
            2: [11, 12, 13, 14, 15],
            3: [16, 17, 18, 19, 20],
            4: [21, 22, 23, 24, 25]
        })

        result = readmatrix(self.test_file, 'A1:E5')
        assert_frame_equal(result, expected_output)

    def test_read_partial_range(self):
        """
        Test reading a smaller range (A2:C4) from the Excel file.
        """
        expected_output = pd.DataFrame({
            0: [2, 3, 4],
            1: [7, 8, 9],
            2: [12, 13, 14]
        }, index=[1, 2, 3])  # Adjust index to match DataFrame structure

        result = readmatrix(self.test_file, 'A2:C4')
        assert_frame_equal(result, expected_output)

    def test_file_not_found(self):
        """
        Test reading from a file that does not exist.
        """
        with self.assertRaises(FileNotFoundError):
            readmatrix('non_existent_file.xlsx', 'A1:B2')

    def tearDown(self):
        """
        Remove the temporary Excel file after testing.
        """
        if os.path.exists(self.test_file):
            os.remove(self.test_file)


class TestUtilsFunctions(unittest.TestCase):

    def test_linspace(self):
        """Test linspace function"""
        result = linspace(0, 10, 5)
        expected = np.array([0., 2.5, 5., 7.5, 10.])
        assert_array_equal(result, expected)

    def test_meshgrid(self):
        """Test meshgrid function"""
        x = np.array([1, 2, 3])
        y = np.array([4, 5])
        xv, yv = meshgrid(x, y)
        expected_xv = np.array([[1, 2, 3], [1, 2, 3]])
        expected_yv = np.array([[4, 4, 4], [5, 5, 5]])
        assert_array_equal(xv, expected_xv)
        assert_array_equal(yv, expected_yv)

    def test_integral(self):
        """Test integral function for single integrals"""
        f = lambda x: x**2
        result = integral(f, 0, 1)
        expected = 1 / 3
        self.assertAlmostEqual(result, expected)

    def test_integral2(self):
        """Test integral2 function for double integrals"""
        f = lambda y, x: x * y ** 2
        result = integral2(f, 0, 2, 0, 1)
        expected = 2 / 3
        self.assertAlmostEqual(result, expected)

    def test_randperm(self):
        """Test randperm function"""
        n = 5
        result = randperm(n)
        self.assertEqual(len(result), n)
        self.assertTrue(np.array_equal(np.sort(result), np.arange(1, n + 1)))

    def test_randi(self):
        """Test randi function"""
        result = randi(10, 5)
        self.assertEqual(result.shape[0], 5)
        self.assertTrue(np.all(result >= 1))
        self.assertTrue(np.all(result <= 10))

        result = randi([1, 5], 10)
        self.assertTrue(np.all(result >= 1))
        self.assertTrue(np.all(result <= 5))

    def test_rand(self):
        """Test rand function"""
        result = rand(3, 2)
        self.assertEqual(result.shape, (3, 2))
        self.assertTrue(np.all(result >= 0))
        self.assertTrue(np.all(result < 1))

    def test_normrnd(self):
        """Test normrnd function"""
        np.random.seed(0)  # Seed for reproducibility
        mu, sigma = 0, 1
        result = normrnd(mu, sigma, 100)
        self.assertEqual(result.shape[0], 100)
        self.assertAlmostEqual(np.mean(result), mu, delta=0.2)  # Slightly loosened delta
        self.assertAlmostEqual(np.std(result), sigma, delta=0.2)

    def test_chi2rnd(self):
        """Test chi2rnd function"""
        nu = 3
        result = chi2rnd(nu, 100)
        self.assertEqual(result.shape[0], 100)
        self.assertTrue(np.all(result >= 0))
        self.assertAlmostEqual(np.mean(result), nu, delta=0.5)


class TestMissingDataFunctions(unittest.TestCase):

    def setUp(self):
        """Set up test data with missing values"""
        self.df_with_nan = pd.DataFrame({
            'A': [1, 2, np.nan, 4, 5],
            'B': [np.nan, 7, 8, np.nan, 10],
            'C': [11, 12, 13, np.nan, 15]
        })
        self.arr_with_nan = np.array([
            [1, 2, np.nan],
            [4, np.nan, 6],
            [7, 8, 9]
        ])

    def test_rmmissing_dataframe(self):
        """Test rmmissing function with pandas DataFrame"""
        result = rmmissing(self.df_with_nan).reset_index(drop=True)  # Reset index to align with expected DataFrame
        expected = pd.DataFrame({
            'A': [2, 5],
            'B': [7, 10],
            'C': [12, 15]
        })
        assert_frame_equal(result, expected.astype(float))

    def test_rmmissing_array(self):
        """Test rmmissing function with numpy array"""
        result = rmmissing(self.arr_with_nan)
        expected = np.array([[7, 8, 9]])
        assert_array_equal(result, expected)

    def test_nanmean_dataframe(self):
        """Test nanmean function with pandas DataFrame"""
        result = nanmean(self.df_with_nan)
        expected = np.array([3, 8.333333333333334, 12.75])
        assert_array_equal(result, expected)

    def test_nanmean_array(self):
        """Test nanmean function with numpy array"""
        result = nanmean(self.arr_with_nan)
        expected = np.array([4, 5, 7.5])
        assert_array_equal(result, expected)

    def test_fillmissing_with_mean_dataframe(self):
        """Test fillmissing_with_mean function with pandas DataFrame"""
        result = fillmissing_with_mean(self.df_with_nan)
        expected = pd.DataFrame({
            'A': [1, 2, 3, 4, 5],
            'B': [8.333333333333334, 7, 8, 8.333333333333334, 10],
            'C': [11, 12, 13, 12.75, 15]
        })
        assert_frame_equal(result, expected.astype(float))

    def test_fillmissing_with_mean_array(self):
        """Test fillmissing_with_mean function with numpy array"""
        result = fillmissing_with_mean(self.arr_with_nan)
        expected = np.array([
            [1, 2, 7.5],
            [4, 5, 6],
            [7, 8, 9]
        ])
        assert_array_equal(result, expected)


if __name__ == '__main__':
    unittest.main()
