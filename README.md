# Statistical Methods for Data Science (statsToolkit)
The project is sponsored by **Malmö Universitet** developed by Eng. Marco Schivo and Eng. Alberto Biscalchin under the supervision of Associete Professor Yuanji Cheng and is released under the **MIT License**. It is open source and available for anyone to use and contribute to.

Code course: MA660E

---

# Descriptive Statistics

This module contains basic descriptive statistics functions that allow users to perform statistical analysis on numerical datasets. The functions are designed for flexibility and ease of use, and they provide essential statistical metrics such as mean, median, range, variance, standard deviation, and quantiles.

## Functions Overview

### 1. `mean(X)`
Calculates the arithmetic mean (average) of a list of numbers.

#### Example Usage:
```python
from statstoolkit.descriptive_statistics import mean

data = [1, 2, 3, 4, 5]
print(mean(data))  # Output: 3.0
```

### 2. `median(X)`
Calculates the median of a list of numbers. The median is the middle value that separates the higher half from the lower half of the dataset.

#### Example Usage:
```python
from statstoolkit.descriptive_statistics import median

data = [1, 2, 3, 4, 5]
print(median(data))  # Output: 3
```

### 3. `range_(X)`
Calculates the range, which is the difference between the maximum and minimum values in the dataset.

#### Example Usage:
```python
from statstoolkit.descriptive_statistics import range_

data = [1, 2, 3, 4, 5]
print(range_(data))  # Output: 4
```

### 4. `var(X, ddof=0)`
Calculates the variance of the dataset. Variance measures the spread of the data from the mean. You can calculate both population variance (`ddof=0`) or sample variance (`ddof=1`).

#### Example Usage:
```python
from statstoolkit.descriptive_statistics import var

data = [1, 2, 3, 4, 5]
print(var(data))  # Output: 2.0  (Population variance)
print(var(data, ddof=1))  # Output: 2.5  (Sample variance)
```

### 5. `std(X, ddof=0)`
Calculates the standard deviation, which is the square root of the variance. It indicates how much the data varies from the mean.

#### Example Usage:
```python
from statstoolkit.descriptive_statistics import std

data = [1, 2, 3, 4, 5]
print(std(data))  # Output: 1.4142135623730951  (Population standard deviation)
print(std(data, ddof=1))  # Output: 1.5811388300841898  (Sample standard deviation)
```

### 6. `quantile(X, Q)`
Calculates the quantile, which is the value below which a given percentage of the data falls. For example, the 0.25 quantile is the first quartile (25th percentile).

#### Example Usage:
```python
from statstoolkit.descriptive_statistics import quantile

data = [1, 2, 3, 4, 5]
print(quantile(data, 0.25))  # Output: 2.0 (25th percentile)
print(quantile(data, 0.5))   # Output: 3.0 (Median)
print(quantile(data, 0.75))  # Output: 4.0 (75th percentile)
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

