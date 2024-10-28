from statstoolkit.statistics import std
import numpy as np

# Test 1: Basic vector with default behavior (N-1 normalization)
A1 = [10, 15, 20, 25, 30]

# Test 2: 2D matrix with different dimensions
A2 = np.array([[100, 200, 300], [400, 500, 600]])

# Test 3: Multidimensional array (3D) with default behavior
A3 = np.array([[[2, 4], [-2, 1]], [[9, 13], [-5, 7]], [[4, 4], [8, -3]]])

# Test 4: Weighted standard deviation (weight vector)
A4 = [5, 10, 15, 20, 25]
w4 = [0.1, 0.2, 0.3, 0.4, 0.5]

# Test 5: Population standard deviation (N normalization)
A5 = [50, 55, 60, 65, 70]

# Test 6: Missing values, ignore NaN
A6 = [10, np.nan, 20, np.nan, 30]

# Test 7: Single dimension (dim) in 3D array
A7 = np.array([[[2, 4, 6], [-2, 0, 2]], [[3, 7, 11], [-1, 2, 5]], [[4, 8, 12], [-3, 1, 3]]])

# Test 8: Vector with no variance (constant values)
A8 = [1000, 1000, 1000, 1000, 1000]

# Test 9: Matrix with multiple dimensions (vecdim) on 2D array
A9 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# Test 10: Large array of random values
np.random.seed(0)
A10 = np.random.randn(1000) * 10 + 50

# Run tests
print("Test 1:", std(A1))
print("Test 2:", std(A2, dim=0))
print("Test 3:", std(A3, vecdim=[0, 1]))
print("Test 4:", std(A4, w=w4))
print("Test 5:", std(A5, w=1))
print("Test 6:", std(A6, missingflag="omitmissing"))
print("Test 7:", std(A7, dim=2))
print("Test 8:", std(A8))
print("Test 9:", std(A9, vecdim=[0, 1]))
print("Test 10:", std(A10))