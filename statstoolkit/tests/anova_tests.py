import unittest
import pandas as pd
import numpy as np
from statstoolkit.statistics import anova

class TestAnovaFunction(unittest.TestCase):

    def test_one_way_anova(self):
        y = np.array([[5, 6, 7], [4, 5, 6], [8, 7, 9]])
        result = anova(y=y)
        self.assertIn('C(factor)', result.index)

    def test_two_way_anova_with_factors(self):
        y = [5, 6, 7, 4, 5, 6, 8, 7, 9]
        factor_1 = ['A', 'A', 'A', 'B', 'B', 'B', 'C', 'C', 'C']
        factor_2 = ['X', 'X', 'Y', 'Y', 'X', 'X', 'Y', 'Y', 'X']
        result = anova(y=y, factors=[factor_1, factor_2])

        # Dynamically check factor columns based on their actual labels in the result
        factor_names = [name for name in result.index if name.startswith("C(factor_")]
        self.assertEqual(len(factor_names), 2)  # Check we have exactly 2 factors
        self.assertIn("C(factor_0)", factor_names)
        self.assertIn("C(factor_1)", factor_names)

    def test_single_factor_level(self):
        y = [5, 6, 7]
        factor = ['A', 'A', 'A']  # Only one level in factor
        with self.assertRaises(ValueError):
            anova(y=y, factors=[factor])

    def test_formula_based_anova(self):
        data = pd.DataFrame({
            'y': [23, 25, 20, 21, 19, 18, 22, 24],
            'A': ['High', 'High', 'Low', 'Low', 'High', 'High', 'Low', 'Low'],
            'B': ['Type1', 'Type2', 'Type1', 'Type2', 'Type1', 'Type2', 'Type1', 'Type2']
        })
        result = anova(y='y', data=data, formula='y ~ A + B + A:B')
        self.assertIn('A', result.index)
        self.assertIn('B', result.index)
        self.assertIn('A:B', result.index)

    def test_table_based_anova_with_response_varname(self):
        data = pd.DataFrame({
            'response': [20, 22, 19, 21, 18, 24, 23, 17],
            'factor1': ['A', 'A', 'B', 'B', 'A', 'A', 'B', 'B'],
            'factor2': ['X', 'Y', 'X', 'Y', 'X', 'Y', 'X', 'Y']
        })
        result = anova(y='response', data=data, formula='response ~ factor1 + factor2')
        self.assertIn('factor1', result.index)
        self.assertIn('factor2', result.index)

    def test_type_iii_sum_of_squares(self):
        data = pd.DataFrame({
            'y': [18, 22, 19, 21, 20, 23, 24, 17],
            'A': ['High', 'High', 'Low', 'Low', 'High', 'High', 'Low', 'Low'],
            'B': ['Type1', 'Type2', 'Type1', 'Type2', 'Type1', 'Type2', 'Type1', 'Type2']
        })
        result = anova(y='y', data=data, formula='y ~ A + B + A:B', sum_of_squares='type III')
        self.assertIn('A', result.index)
        self.assertIn('B', result.index)
        self.assertIn('A:B', result.index)

    def test_invalid_sum_of_squares_type(self):
        data = pd.DataFrame({
            'y': [18, 22, 19, 21, 20, 23, 24, 17],
            'A': ['High', 'High', 'Low', 'Low', 'High', 'High', 'Low', 'Low'],
            'B': ['Type1', 'Type2', 'Type1', 'Type2', 'Type1', 'Type2', 'Type1', 'Type2']
        })
        with self.assertRaises(ValueError):
            anova(y='y', data=data, formula='y ~ A + B + A:B', sum_of_squares='type IV')

    def test_empty_response(self):
        y = np.array([])
        with self.assertRaises(ValueError):
            anova(y=y)

    def test_missing_intercept_in_formula(self):
        data = pd.DataFrame({
            'y': [23, 25, 20, 21, 19, 18, 22, 24],
            'A': ['High', 'High', 'Low', 'Low', 'High', 'High', 'Low', 'Low'],
            'B': ['Type1', 'Type2', 'Type1', 'Type2', 'Type1', 'Type2', 'Type1', 'Type2']
        })
        result = anova(y='y', data=data, formula='y ~ A + B - 1')
        self.assertIn('A', result.index)
        self.assertIn('B', result.index)

    def test_unbalanced_factors(self):
        y = np.array([10, 12, 15, 13])
        factor1 = pd.Series(['A', 'A', 'B'])
        with self.assertRaises(ValueError):
            anova(factors=[factor1], y=y)

    def test_response_varname_invalid(self):
        data = pd.DataFrame({
            'x': [20, 22, 19, 21],
            'factor': ['A', 'B', 'A', 'B']
        })
        with self.assertRaises(ValueError):
            anova(y='nonexistent_column', data=data)

if __name__ == '__main__':
    unittest.main()
