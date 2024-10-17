import unittest
from statstoolkit.statistics import mean, median, range_, var, std, quantile


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


if __name__ == '__main__':
    unittest.main()
