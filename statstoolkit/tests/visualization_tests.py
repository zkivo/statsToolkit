import unittest
import matplotlib.pyplot as plt
import numpy as np
from statstoolkit.visualization import (
    bar_chart, pie_chart, histogram, boxplot, scatterplot, plot_regression_surface
)


class TestVisualizationFunctions(unittest.TestCase):

    def setUp(self):
        """
        Set up test data for the visualizations.
        """
        self.x = [1, 2, 3, 4]
        self.y = [10, 20, 30, 40]
        self.labels = ['A', 'B', 'C', 'D']
        self.sizes = [10, 20, 30, 40]
        self.mpg = [15, 18, 21, 24, 30]
        self.origin = ['USA', 'Japan', 'Europe', 'USA', 'Japan']
        self.z = [100, 200, 300, 400]

        # Set up data for regression visualization tests
        self.weight = np.array([2000, 3000, 4000, 5000])
        self.horsepower = np.array([50, 100, 150, 200])
        self.mpg_values = np.array([35, 25, 15, 10])

        # Regression coefficients
        self.b = [60.7104, -0.0102, -0.1882, 0.0000]  # Example coefficients

    def test_bar_chart(self):
        """
        Test that the bar chart is created without error.
        """
        try:
            bar_chart(self.labels, self.y, title="Test Bar Chart", xlabel="Category", ylabel="Value")
        except Exception as e:
            self.fail(f"bar_chart raised an exception: {e}")

    def test_pie_chart(self):
        """
        Test that the pie chart is created without error.
        """
        try:
            pie_chart(self.sizes, labels=self.labels, title="Test Pie Chart")
        except Exception as e:
            self.fail(f"pie_chart raised an exception: {e}")

    def test_histogram(self):
        """
        Test that the histogram is created without error.
        """
        try:
            histogram(self.x, bins=4, title="Test Histogram", xlabel="Value", ylabel="Frequency")
        except Exception as e:
            self.fail(f"histogram raised an exception: {e}")

    def test_boxplot(self):
        """
        Test that the boxplot is created without error.
        """
        try:
            boxplot(self.mpg, title="Test Boxplot", ylabel="MPG")
        except Exception as e:
            self.fail(f"boxplot raised an exception: {e}")

        try:
            boxplot(self.mpg, origin=self.origin, title="Test Boxplot by Origin", xlabel="Origin", ylabel="MPG")
        except Exception as e:
            self.fail(f"boxplot by origin raised an exception: {e}")

    def test_scatterplot_2d(self):
        """
        Test that a 2D scatter plot is created without error.
        """
        try:
            scatterplot(self.x, self.y, symbol="o", title="Test Scatterplot 2D", xlabel="X-axis", ylabel="Y-axis")
        except Exception as e:
            self.fail(f"scatterplot 2D raised an exception: {e}")

    def test_scatterplot_3d(self):
        """
        Test that a 3D-like scatter plot (with a z variable) is created without error.
        """
        try:
            scatterplot(self.x, self.y, self.z, symbol="*", title="Test Scatterplot 3D", xlabel="X-axis",
                        ylabel="Y-axis")
        except Exception as e:
            self.fail(f"scatterplot 3D raised an exception: {e}")

    def test_plot_regression_surface(self):
        """
        Test that the regression surface plot is created without error in 3D.
        """
        try:
            plot_regression_surface(
                x1=self.weight,
                x2=self.horsepower,
                y=self.mpg_values,
                b=self.b  # Coefficients
            )
        except Exception as e:
            self.fail(f"plot_regression_surface raised an exception: {e}")

    def tearDown(self):
        """
        Close all figures after each test to prevent memory issues.
        """
        plt.close('all')


if __name__ == '__main__':
    unittest.main()
