import unittest
from statstoolkit.statistics import mean, median, range_, var, std, quantile, covariance, fitlm
from statstoolkit.statistics import partialcorr
import numpy as np


class TestStatFunctions(unittest.TestCase):

    def test_mean(self):
        self.assertEqual(mean([1, 2, 3, 4, 5]), 3)
        self.assertEqual(mean([10, 20, 30]), 20)
        self.assertAlmostEqual(mean([1.1, 2.2, 3.3]), 2.2)
        with self.assertRaises(ValueError):
            mean([])

    def test_median(self):
        self.assertEqual(median([1, 2, 3, 4, 5]), 3)
        self.assertEqual(median([1, 2, 3, 4]), 2.5)
        self.assertEqual(median([7, 1, 3]), 3)
        with self.assertRaises(ValueError):
            median([])

    def test_range(self):
        self.assertEqual(range_([1, 2, 3, 4, 5]), 4)
        self.assertEqual(range_([10, 20, 30, 40]), 30)
        self.assertEqual(range_([5]), 0)
        with self.assertRaises(ValueError):
            range_([])

    def test_var(self):
        self.assertEqual(var([1, 2, 3, 4, 5]), 2.0)  # Population variance
        self.assertAlmostEqual(var([1, 2, 3, 4, 5], ddof=1), 2.5)  # Sample variance
        self.assertAlmostEqual(var([2.5, 3.5, 7.5]), 4.666666666666667)
        with self.assertRaises(ValueError):
            var([])

    def test_std(self):
        self.assertAlmostEqual(std([1, 2, 3, 4, 5]), 1.4142135623730951)  # Population std
        self.assertAlmostEqual(std([1, 2, 3, 4, 5], ddof=1), 1.5811388300841898)  # Sample std
        self.assertAlmostEqual(std([2.5, 3.5, 7.5]), 2.160246899469287)
        with self.assertRaises(ValueError):
            std([])

    def test_quantile(self):
        self.assertEqual(quantile([1, 2, 3, 4, 5], 0.25), 2.0)  # First quartile
        self.assertEqual(quantile([1, 2, 3, 4, 5], 0.5), 3.0)  # Median
        self.assertEqual(quantile([1, 2, 3, 4, 5], 0.75), 4.0)  # Third quartile
        self.assertEqual(quantile([7, 1, 3], 0.5), 3.0)  # Median of unsorted list
        with self.assertRaises(ValueError):
            quantile([], 0.5)
        with self.assertRaises(ValueError):
            quantile([1, 2, 3, 4, 5], -0.5)
        with self.assertRaises(ValueError):
            quantile([1, 2, 3, 4, 5], 1.5)

    def test_partialcorr_two_variables(self):
        """Test partial correlation with two variables."""
        data = np.array([[1, 2], [2, 4], [3, 6], [4, 8]])  # Two perfectly collinear variables
        result = partialcorr(data)
        expected = np.array([[1.0, -1.0], [-1.0, 1.0]])  # Correcting the expectation
        np.testing.assert_array_almost_equal(result.to_numpy(), expected, decimal=3)

    def test_partialcorr_three_variables(self):
        """Test partial correlation with three variables."""
        data = np.array([
            [1, 2, 1],
            [2, 3, 2],
            [3, 4, 3],
            [4, 5, 4]
        ])
        result = partialcorr(data)
        expected = np.array([[1.0, -1.0, -1.0], [-1.0, 1.0, -1.0], [-1.0, -1.0, 1.0]])
        np.testing.assert_array_almost_equal(result.to_numpy(), expected, decimal=2)

    def test_partialcorr_with_controlled_variable(self):
        """Test partial correlation while controlling for a third variable."""
        data = np.array([
            [1, 2, 1],
            [2, 3, 1],
            [3, 4, 2],
            [4, 5, 2]
        ])
        result = partialcorr(data)
        expected = np.array([[1.0, -1.0, 0.89], [-1.0, 1.0, 0.89], [0.89, 0.89, 1.0]])
        np.testing.assert_array_almost_equal(result.to_numpy(), expected, decimal=2)


    def test_covariance_two_vectors(self):
        a = [1, 0, -1, 3, 5, -2, 0.5]
        b = [-1, 2, 4, -0.5, 1, 1, 0]
        result = covariance(a, b)
        expected = np.array([[5.7024, -1.5893],
                             [-1.5893, 2.8690]])
        np.testing.assert_array_almost_equal(result, expected, decimal=4)

    def test_covariance_matrix(self):
        a = [1, 0, -1, 3, 5, -2, 0.5]
        b = [-1, 2, 4, -0.5, 1, 1, 0]
        c = [-0.4, 1.2, 0, 3, 2.5, -1, 6]
        X = np.array([a, b, c]).T
        result = covariance(X)
        expected = np.array([
            [5.7024, -1.5893, 2.6012],
            [-1.5893, 2.8690, -1.2821],
            [2.6012, -1.2821, 5.9348]
        ])
        np.testing.assert_array_almost_equal(result, expected, decimal=4)

    def test_covariance_single_vector(self):
        a = [1, 0, -1, 3, 5, -2, 0.5]
        result = covariance(a)
        expected = np.array([[np.var(a, ddof=1)]])  # Expecting a 2D array for consistency
        np.testing.assert_array_almost_equal(result, expected, decimal=4)

    def test_covariance_mismatched_lengths(self):
        a = [1, 0, -1]
        b = [1, 2]
        with self.assertRaises(ValueError):
            covariance(a, b)

    def test_covariance_empty_input(self):
        with self.assertRaises(ValueError):
            covariance([])

    def test_basic_linear_regression(self):
        """Test basic linear regression with a simple case."""
        x = [1, 2, 3, 4, 5, 6, 7]
        y = [1.5, 3, 4.5, 6, 7.5, 9, 10.5]
        result = fitlm(x, y)
        expected_intercept = 0.0
        expected_slope = 1.5
        np.testing.assert_almost_equal(result["Coefficients"].loc["Intercept", "Estimate"], expected_intercept, decimal=3)
        np.testing.assert_almost_equal(result["Coefficients"].iloc[1]["Estimate"], expected_slope, decimal=3)

    def test_insufficient_data(self):
        """Test regression with fewer than 2 data points to ensure it raises ValueError."""
        x = [1]
        y = [2]
        with self.assertRaises(ValueError):
            fitlm(x, y)

    def test_zero_variance_in_x(self):
        """Test regression with zero variance in x to ensure it raises ValueError."""
        x = [2, 2, 2, 2, 2]
        y = [1, 2, 3, 4, 5]
        with self.assertRaises(ValueError):
            fitlm(x, y)

    def test_zero_variance_in_y(self):
        """Test regression with zero variance in y to ensure it raises ValueError."""
        x = [1, 2, 3, 4, 5]
        y = [3, 3, 3, 3, 3]
        with self.assertRaises(ValueError):
            fitlm(x, y)

    def test_perfectly_collinear_data(self):
        """Test with perfectly collinear data to check for expected behavior in slope and R-squared."""
        x = [1, 2, 3, 4, 5]
        y = [2, 4, 6, 8, 10]
        result = fitlm(x, y)
        expected_slope = 2.0
        expected_intercept = 0.0
        np.testing.assert_almost_equal(result["Coefficients"].loc["Intercept", "Estimate"], expected_intercept, decimal=3)
        np.testing.assert_almost_equal(result["Coefficients"].iloc[1]["Estimate"], expected_slope, decimal=3)
        self.assertAlmostEqual(result["R-squared"], 1.0, places=3)

    def test_horizontal_line(self):
        """Test with a horizontal line where y has the same value for all x."""
        x = [1, 2, 3, 4, 5]
        y = [4, 4, 4, 4, 4]
        with self.assertRaises(ValueError):
            fitlm(x, y)

    def test_identical_points(self):
        """Test with identical points to ensure it raises ValueError due to lack of variance."""
        x = [3, 3, 3, 3]
        y = [5, 5, 5, 5]
        with self.assertRaises(ValueError):
            fitlm(x, y)

    def test_multiple_observations(self):
        """Test standard linear regression with realistic data and verify computed values."""
        x = [10, 20, 30, 40, 50, 60]
        y = [15, 25, 35, 45, 55, 65]
        result = fitlm(x, y)
        expected_intercept = 5.0
        expected_slope = 1.0
        np.testing.assert_almost_equal(result["Coefficients"].loc["Intercept", "Estimate"], expected_intercept, decimal=3)
        np.testing.assert_almost_equal(result["Coefficients"].iloc[1]["Estimate"], expected_slope, decimal=3)
        self.assertAlmostEqual(result["R-squared"], 1.0, places=3)

if __name__ == '__main__':
    unittest.main()
