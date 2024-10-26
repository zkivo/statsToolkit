import unittest
import matplotlib.pyplot as plt
import numpy as np
from statstoolkit.visualization import plot_regression_surface
from statstoolkit.statistics import regress


class TestVisualizationFunctions(unittest.TestCase):

    def setUp(self):
        """
        Set up realistic test data for the regression and plotting functions.
        """
        # Generate synthetic data for a realistic multiple regression test
        np.random.seed(0)
        self.weight = np.random.uniform(2000, 5000, 50)  # Predictor variable 1 (e.g., Weight)
        self.horsepower = np.random.uniform(50, 250, 50)  # Predictor variable 2 (e.g., Horsepower)

        # Generate a response variable (e.g., MPG) with some noise
        self.mpg = 60 + (-0.01 * self.weight) + (-0.18 * self.horsepower) + \
                   (0.00005 * self.weight * self.horsepower) + np.random.normal(0, 5, 50)

        # Prepare the design matrix with interaction terms
        self.X = np.column_stack(
            (np.ones(len(self.weight)), self.weight, self.horsepower, self.weight * self.horsepower))

    def test_regress_and_plot_surface_combined(self):
        """
        Test the regress and plot_regression_surface functions together in a realistic scenario.
        """
        # Perform regression
        try:
            b, _, _, _, _ = regress(self.mpg, self.X)
        except Exception as e:
            self.fail(f"regress function raised an exception: {e}")

        # Check if coefficients are in expected range (not overly large/small)
        expected_range = [-1e2, 1e2]
        for coef in b:
            self.assertGreaterEqual(coef, expected_range[0], "Coefficient too small!")
            self.assertLessEqual(coef, expected_range[1], "Coefficient too large!")

        # Plot the regression surface using the obtained coefficients
        try:
            plot_regression_surface(self.weight, self.horsepower, self.mpg, b)
        except Exception as e:
            self.fail(f"plot_regression_surface raised an exception when combined with regress: {e}")

    def tearDown(self):
        """
        Close all figures after each test to prevent memory issues.
        """
        plt.close('all')


if __name__ == '__main__':
    unittest.main()
