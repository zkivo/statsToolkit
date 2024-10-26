import unittest
import numpy as np
import pandas as pd
from statstoolkit.statistics import kruskalwallis

class TestKruskalWallisFunction(unittest.TestCase):

    def test_single_column_input(self):
        x = [5, 6, 7, 8, 9]
        with self.assertRaises(ValueError):
            kruskalwallis(x)

    def test_multiple_columns(self):
        x = np.array([[5, 6, 7], [4, 5, 6], [8, 7, 9]])
        p, tbl, stats = kruskalwallis(x)
        self.assertIsInstance(p, float)
        self.assertIsInstance(tbl, pd.DataFrame)
        self.assertIn('H', tbl.columns)
        self.assertIn('p-value', tbl.columns)

    def test_group_labels(self):
        x = [5, 6, 7, 8, 9, 10, 11, 12, 13]
        group = ['A', 'A', 'A', 'B', 'B', 'B', 'C', 'C', 'C']
        p, tbl, stats = kruskalwallis(x, group=group)
        self.assertIsInstance(p, float)
        self.assertFalse(np.isnan(p))  # Check that p-value is a valid number
        self.assertIn('p-value', tbl.columns)

    def test_displayopt_true(self):
        x = [5, 6, 7, 8, 9, 10, 11, 12, 13]
        group = ['A', 'A', 'A', 'B', 'B', 'B', 'C', 'C', 'C']
        try:
            p, tbl, stats = kruskalwallis(x, group=group, displayopt=True)
            self.assertIsInstance(p, float)
            self.assertIsInstance(tbl, pd.DataFrame)
        except Exception as e:
            self.fail(f"kruskalwallis raised an exception with displayopt=True: {e}")

    def test_equal_groups(self):
        x = [5, 5, 5, 5, 5, 5, 5, 5, 5]
        group = ['A', 'A', 'A', 'B', 'B', 'B', 'C', 'C', 'C']
        with self.assertRaises(ValueError):
            kruskalwallis(x, group=group)

    def test_different_distributions(self):
        x = [1, 2, 3, 7, 8, 9, 15, 16, 17]
        group = ['A', 'A', 'A', 'B', 'B', 'B', 'C', 'C', 'C']
        p, tbl, stats = kruskalwallis(x, group=group)
        self.assertLess(p, 0.05)

    def test_output_structure(self):
        x = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        group = ['A', 'A', 'A', 'B', 'B', 'B', 'C', 'C', 'C']
        p, tbl, stats = kruskalwallis(x, group=group)
        self.assertIn('Source', tbl.columns)
        self.assertIn('H', tbl.columns)
        self.assertIn('p-value', tbl.columns)
        self.assertIn('test_statistic', stats)
        self.assertIn('p_value', stats)
        self.assertIn('df', stats)

    def test_empty_input(self):
        x = []
        group = []
        with self.assertRaises(ValueError):
            kruskalwallis(x, group=group)

    def test_mismatched_x_and_group_length(self):
        x = [5, 6, 7, 8]
        group = ['A', 'B']
        with self.assertRaises(ValueError):
            kruskalwallis(x, group=group)

if __name__ == '__main__':
    unittest.main()
