import unittest
from statstoolkit.statistics import mean, median, range_, var, std, quantile, fitlm, cov, regress, ttest, ttest2
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

    def test_one_sample_default_mean(self):
        """Test one-sample t-test with default mean (0)."""
        x = [2.1, 2.5, 2.8, 3.2, 2.7]
        h, p, ci, stats = ttest(x)
        expected_h = 1  # Null hypothesis rejected
        self.assertEqual(h, expected_h)
        self.assertTrue(0 < p < 0.05)  # Significance level at 5%
        self.assertTrue(ci[0] < np.mean(x) < ci[1])  # CI should contain mean

    def test_one_sample_non_zero_mean(self):
        """Test one-sample t-test with non-zero mean (m=2.5)."""
        x = [2.1, 2.5, 2.8, 3.2, 2.7]
        h, p, ci, stats = ttest(x, m=2.5)
        self.assertIsInstance(h, int)
        self.assertIsInstance(p, float)
        self.assertTrue(ci[0] < np.mean(x) - 2.5 < ci[1])

    def test_one_sample_one_sided_greater(self):
        """Test one-sample one-sided t-test with alternative='greater'."""
        x = [2.5, 2.7, 3.1, 2.9, 3.0]  # Mean above the null hypothesis mean of 2.0
        m = 2.0
        expected_h = 1  # We expect to reject the null hypothesis

        # Run the test
        h, p, ci, stats = ttest(x, m=m, alternative='greater')

        self.assertEqual(h, expected_h)

    def test_one_sample_one_sided_less(self):
        """Test one-sample t-test with one-sided alternative (less)."""
        x = [1.1, 1.5, 1.8, 1.3, 1.7]
        h, p, ci, stats = ttest(x, m=2.0, alternative='less')
        expected_h = 1  # Null hypothesis rejected
        self.assertEqual(h, expected_h)
        self.assertTrue(0 < p < 0.05)  # One-sided p-value should be less than 0.05
        self.assertTrue(ci[0] < np.mean(x) - 2.0 < ci[1])

    def test_paired_sample(self):
        """Test paired-sample t-test with two related samples."""
        x = [2.1, 2.5, 2.8, 3.0, 2.7]
        y = [1.9, 2.3, 2.6, 2.8, 2.5]
        h, p, ci, stats = ttest(x, y=y)
        expected_h = 1  # Null hypothesis rejected
        self.assertEqual(h, expected_h)
        self.assertTrue(0 < p < 0.05)
        self.assertTrue(ci[0] < np.mean(np.array(x) - np.array(y)) < ci[1])

    def test_different_confidence_level(self):
        """Test one-sample t-test with a higher confidence level (99%)."""
        x = [2.1, 2.5, 2.8, 3.2, 2.7]
        h, p, ci, stats = ttest(x, alpha=0.01)  # 99% confidence level
        expected_h = 1  # Null hypothesis rejected
        self.assertEqual(h, expected_h)
        self.assertTrue(ci[0] < np.mean(x) < ci[1])

    def test_no_significant_difference(self):
        """Test one-sample t-test where null hypothesis is not rejected."""
        x = [0.1, -0.2, 0.05, -0.1, 0.15]
        h, p, ci, stats = ttest(x)
        expected_h = 0  # Null hypothesis not rejected
        self.assertEqual(h, expected_h)
        self.assertTrue(p > 0.05)

    def test_stats_output(self):
        """Test that the stats dictionary contains t-statistic and df."""
        x = [2.1, 2.5, 2.8, 3.2, 2.7]
        h, p, ci, stats = ttest(x)
        self.assertIn("t_stat", stats)
        self.assertIn("df", stats)
        self.assertIsInstance(stats["t_stat"], float)
        self.assertIsInstance(stats["df"], int)

    def test_invalid_alternative(self):
        """Test that invalid alternative hypothesis raises ValueError."""
        x = [2.1, 2.5, 2.8, 3.2, 2.7]
        with self.assertRaises(ValueError):
            ttest(x, alternative="invalid_option")

    def test_invalid_alpha(self):
        """Test that invalid alpha raises ValueError."""
        x = [2.1, 2.5, 2.8, 3.2, 2.7]
        with self.assertRaises(ValueError):
            ttest(x, alpha=1.5)

    def test_empty_sample(self):
        """Test that empty sample raises ValueError."""
        x = []
        with self.assertRaises(ValueError):
            ttest(x)

    def test_mismatched_lengths(self):
        """Test that mismatched lengths for paired-sample raises ValueError."""
        x = [2.1, 2.5, 2.8]
        y = [1.9, 2.3]
        with self.assertRaises(ValueError):
            ttest(x, y=y)

    def test_two_sample_equal_var_default(self):
        x = [2, 4, 6, 8, 10]
        y = [1, 3, 5, 7, 9]
        h, p, ci, stats = ttest2(x, y)
        self.assertEqual(h, 0)
        self.assertGreater(p, 0.05)

    def test_two_sample_unequal_var(self):
        x = [15, 18, 21, 24, 30]
        y = [22, 25, 29, 32, 35]
        h, p, ci, stats = ttest2(x, y, equal_var=False)
        self.assertEqual(h, 0)
        self.assertGreater(p, 0.05)

    def test_two_sample_custom_alpha(self):
        x = [5, 10, 15, 20, 25]
        y = [8, 12, 18, 22, 27]
        h, p, ci, stats = ttest2(x, y, alpha=0.01)
        self.assertEqual(h, 0)
        self.assertGreater(p, 0.01)

    def test_two_sample_alternative_greater(self):
        x = [10, 12, 14, 16, 18]
        y = [6, 7, 8, 9, 10]
        h, p, ci, stats = ttest2(x, y, alternative='greater')
        self.assertEqual(h, 1)
        self.assertLess(p, 0.05)

    def test_two_sample_alternative_less(self):
        x = [6, 7, 8, 9, 10]
        y = [10, 12, 14, 16, 18]
        h, p, ci, stats = ttest2(x, y, alternative='less')
        self.assertEqual(h, 1)
        self.assertLess(p, 0.05)

    def test2_invalid_alpha(self):
        x = [1, 2, 3]
        y = [4, 5, 6]
        with self.assertRaises(ValueError):
            ttest2(x, y, alpha=1.5)

    def test2_invalid_alternative(self):
        x = [1, 2, 3]
        y = [4, 5, 6]
        with self.assertRaises(ValueError):
            ttest2(x, y, alternative='invalid_option')

    def test2_empty_sample(self):
        x = []
        y = [1, 2, 3]
        with self.assertRaises(ValueError):
            ttest2(x, y)

    def test2_confidence_interval_structure(self):
        x = [5, 6, 7, 8, 9]
        y = [10, 11, 12, 13, 14]
        _, _, ci, _ = ttest2(x, y)
        self.assertEqual(len(ci), 2)
        self.assertLess(ci[0], ci[1])

    def test2_statistics_output_structure(self):
        x = [5, 10, 15]
        y = [20, 25, 30]
        _, _, _, stats = ttest2(x, y)
        self.assertIn("t_stat", stats)
        self.assertIn("df", stats)
        self.assertIsInstance(stats["t_stat"], float)
        self.assertIsInstance(stats["df"], float)

if __name__ == '__main__':
    unittest.main()
