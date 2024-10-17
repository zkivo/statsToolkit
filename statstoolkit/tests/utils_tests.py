import unittest
import pandas as pd
import os
from pandas.testing import assert_frame_equal
from ..utils import readmatrix  # Import the function to be tested


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
        self.data.to_excel(self.test_file, index=False)

    def test_read_specific_range(self):
        """
        Test reading a specific range (A1:E5) from the Excel file.
        """
        expected_output = pd.DataFrame({
            'A': [1, 2, 3, 4, 5],
            'B': [6, 7, 8, 9, 10],
            'C': [11, 12, 13, 14, 15],
            'D': [16, 17, 18, 19, 20],
            'E': [21, 22, 23, 24, 25]
        })

        result = readmatrix(self.test_file, 'A:E')
        assert_frame_equal(result, expected_output)

    def test_read_partial_range(self):
        """
        Test reading a smaller range (A2:C4) from the Excel file.
        """
        expected_output = pd.DataFrame({
            'A': [2, 3, 4],
            'B': [7, 8, 9],
            'C': [12, 13, 14]
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


if __name__ == '__main__':
    unittest.main()
