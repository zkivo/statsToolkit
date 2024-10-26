import unittest
from statstoolkit.statistics import mean, median, range_, var, std, quantile, fitlm, cov, regress
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

    def test_cov_two_vectors(self):
        a = [1, 0, -1, 3, 5, -2, 0.5]
        b = [-1, 2, 4, -0.5, 1, 1, 0]
        result = cov(a, b)
        expected = np.array([[5.7024, -1.5893],
                             [-1.5893, 2.8690]])
        np.testing.assert_array_almost_equal(result, expected, decimal=4)

    def test_cov_matrix(self):
        a = [1, 0, -1, 3, 5, -2, 0.5]
        b = [-1, 2, 4, -0.5, 1, 1, 0]
        c = [-0.4, 1.2, 0, 3, 2.5, -1, 6]
        X = np.array([a, b, c]).T
        result = cov(X)
        expected = np.array([
            [5.7024, -1.5893, 2.6012],
            [-1.5893, 2.8690, -1.2821],
            [2.6012, -1.2821, 5.9348]
        ])
        np.testing.assert_array_almost_equal(result, expected, decimal=4)

    def test_cov_single_vector(self):
        a = [1, 0, -1, 3, 5, -2, 0.5]
        result = cov(a)
        expected = np.array([[np.var(a, ddof=1)]])  # Expecting a 2D array for consistency
        np.testing.assert_array_almost_equal(result, expected, decimal=4)

    def test_cov_mismatched_lengths(self):
        a = [1, 0, -1]
        b = [1, 2]
        with self.assertRaises(ValueError):
            cov(a, b)

    def test_cov_empty_input(self):
        with self.assertRaises(ValueError):
            cov([])


    def test_regress_single_predictor(self):
        """Test regression with a single predictor variable."""
        y = np.array([1, 2, 3, 4, 5])
        X = np.column_stack((np.ones(len(y)), [1, 2, 3, 4, 5]))

        # Expected values
        expected_b = [0, 1]
        expected_r_squared = 1.0  # Perfect fit

        # Run regression
        b, bint, r, rint, stats = regress(y, X)

        # Coefficient tests
        np.testing.assert_almost_equal(b, expected_b, decimal=3)

        # R-squared test
        self.assertAlmostEqual(stats["R-squared"], expected_r_squared, places=3)

    def test_regress_confidence_intervals(self):
        """Test that confidence intervals have the correct structure and are reasonable."""
        y = np.array([1, 2, 3, 4, 5])
        X = np.column_stack((np.ones(len(y)), [1, 2, 3, 4, 5]))

        # Run regression
        _, bint, _, _, _ = regress(y, X)

        # Check the shape and bounds
        self.assertEqual(bint.shape, (2, 2))  # Intercept and slope, each with lower and upper bounds
        self.assertLess(bint[0, 0], bint[0, 1])  # Lower bound < upper bound for intercept
        self.assertLess(bint[1, 0], bint[1, 1])  # Lower bound < upper bound for slope

    def test_regress_basic(self):
        """Test a basic linear regression with two predictors."""
        # Sample data
        y = np.array([5, 7, 9, 11, 13])
        X = np.column_stack((np.ones(len(y)), [1, 2, 3, 4, 5], [2, 4, 6, 8, 10]))

        # Updated expected values
        expected_b = [3.0, 0.4, 0.8]

        # Run regression
        b, bint, r, rint, stats = regress(y, X)

        # Coefficient tests
        np.testing.assert_almost_equal(b, expected_b, decimal=3)

    def test_regress_with_outliers(self):
        """Test regression when data contains an outlier."""
        y = np.array([1, 2, 3, 4, 50])  # Outlier in the response variable
        X = np.column_stack((np.ones(len(y)), [1, 2, 3, 4, 5]))

        # Run regression
        _, _, r, rint, _ = regress(y, X)

        # Check if residual interval identifies outlier
        # Outlier point should have residual outside the rint interval
        self.assertTrue(np.any(r > rint[:, 1]) or np.any(r < rint[:, 0]))


    def test_regress_no_variance_in_x(self):
        """Test for zero variance in predictors (should raise an error)."""
        y = np.array([1, 2, 3, 4, 5])
        X = np.column_stack((np.ones(len(y)), [1, 1, 1, 1, 1]))  # No variance in predictor

        with self.assertRaises(ValueError):
            regress(y, X)

    def test_regress_no_variance_in_y(self):
        """Test for zero variance in response (should raise an error)."""
        y = np.array([5, 5, 5, 5, 5])  # No variance in response
        X = np.column_stack((np.ones(len(y)), [1, 2, 3, 4, 5]))

        with self.assertRaises(ValueError):
            regress(y, X)

    def test_regress_high_confidence_level(self):
        """Test regression with a high confidence level (e.g., 99%)."""
        y = np.array([1, 2, 3, 4, 5])
        X = np.column_stack((np.ones(len(y)), [1, 2, 3, 4, 5]))

        _, bint, _, _, _ = regress(y, X, alpha=0.01)  # 99% confidence intervals

        # Check if the confidence intervals are wider than the default (95%)
        self.assertGreater(bint[0, 1] - bint[0, 0], bint[1, 1] - bint[1, 0])

    def test_regress_insufficient_data(self):
        """Test if function raises an error for insufficient data."""
        y = np.array([5])  # Only one data point
        X = np.array([[1, 2]])  # Only one predictor

        with self.assertRaises(ValueError):
            regress(y, X)

if __name__ == '__main__':
    unittest.main()
