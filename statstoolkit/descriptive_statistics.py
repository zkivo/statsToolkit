import math


def mean(X):
    """
    Calculate the arithmetic mean (average) of a list of numbers.

    Parameters
    ----------
    X : list or array-like
        A list of numerical values.

    Returns
    -------
    float
        The arithmetic mean of the input values.

    Raises
    ------
    ValueError
        If the input list is empty.

    Examples
    --------
    >>> mean([1, 2, 3, 4, 5])
    3.0
    """
    if len(X) == 0:
        raise ValueError("The input list cannot be empty.")
    return sum(X) / len(X)


def median(X):
    """
    Calculate the median of a list of numbers.

    The median is the value separating the higher half from the lower half
    of the data.

    Parameters
    ----------
    X : list or array-like
        A list of numerical values.

    Returns
    -------
    float
        The median of the input values.

    Raises
    ------
    ValueError
        If the input list is empty.

    Examples
    --------
    >>> median([1, 3, 5, 7])
    4.0
    >>> median([1, 2, 3, 4, 5])
    3
    """
    if len(X) == 0:
        raise ValueError("The input list cannot be empty.")
    sorted_X = sorted(X)
    n = len(X)
    mid = n // 2
    if n % 2 == 0:
        return (sorted_X[mid - 1] + sorted_X[mid]) / 2
    else:
        return sorted_X[mid]


def range_(X):
    """
    Calculate the range of a list of numbers.

    The range is the difference between the maximum and minimum values in the list.

    Parameters
    ----------
    X : list or array-like
        A list of numerical values.

    Returns
    -------
    float
        The range of the input values.

    Raises
    ------
    ValueError
        If the input list is empty.

    Examples
    --------
    >>> range_([1, 2, 3, 4, 5])
    4
    """
    if len(X) == 0:
        raise ValueError("The input list cannot be empty.")
    return max(X) - min(X)


def var(X, ddof=0):
    """
    Calculate the variance of a list of numbers.

    Variance measures the spread of the data from the mean. The formula can adjust for
    population or sample variance depending on the `ddof` parameter.

    Parameters
    ----------
    X : list or array-like
        A list of numerical values.
    ddof : int, optional
        Delta Degrees of Freedom (default is 0 for population variance, set to 1 for sample variance).

    Returns
    -------
    float
        The variance of the input values.

    Raises
    ------
    ValueError
        If the input list is empty.

    Examples
    --------
    >>> var([1, 2, 3, 4, 5])
    2.0
    >>> var([1, 2, 3, 4, 5], ddof=1)
    2.5
    """
    if len(X) == 0:
        raise ValueError("The input list cannot be empty.")
    m = mean(X)
    return sum((x - m) ** 2 for x in X) / (len(X) - ddof)


def std(X, ddof=0):
    """
    Calculate the standard deviation of a list of numbers.

    Standard deviation is the square root of the variance and indicates the amount of variation in the data.

    Parameters
    ----------
    X : list or array-like
        A list of numerical values.
    ddof : int, optional
        Delta Degrees of Freedom (default is 0 for population standard deviation, set to 1 for sample standard deviation).

    Returns
    -------
    float
        The standard deviation of the input values.

    Raises
    ------
    ValueError
        If the input list is empty.

    Examples
    --------
    >>> std([1, 2, 3, 4, 5])
    1.4142135623730951
    >>> std([1, 2, 3, 4, 5], ddof=1)
    1.5811388300841898
    """
    return math.sqrt(var(X, ddof=ddof))


def quantile(X, Q):
    """
    Calculate the quantile of a list of numbers.

    A quantile is a value below which a given percentage of the data falls.
    For example, the 0.25 quantile is the first quartile (25th percentile).

    Parameters
    ----------
    X : list or array-like
        A list of numerical values.
    Q : float
        The quantile to compute, a number between 0 and 1.

    Returns
    -------
    float
        The Q-th quantile of the input values.

    Raises
    ------
    ValueError
        If the input list is empty.
    ValueError
        If Q is not between 0 and 1.

    Examples
    --------
    >>> quantile([1, 2, 3, 4, 5], 0.25)
    2.0
    >>> quantile([1, 2, 3, 4, 5], 0.5)
    3.0
    >>> quantile([1, 2, 3, 4, 5], 0.75)
    4.0
    """
    if len(X) == 0:
        raise ValueError("The input list cannot be empty.")
    if not 0 <= Q <= 1:
        raise ValueError("Q must be between 0 and 1.")

    sorted_X = sorted(X)
    pos = (len(X) - 1) * Q
    lower = math.floor(pos)
    upper = math.ceil(pos)

    if lower == upper:
        return sorted_X[int(pos)]
    else:
        lower_value = sorted_X[lower]
        upper_value = sorted_X[upper]
        return lower_value + (upper_value - lower_value) * (pos - lower)


