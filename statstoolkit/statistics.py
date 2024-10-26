import scipy.stats as stats 
import math
import pandas as pd
import numpy as np
from numpy.linalg import inv, LinAlgError
import pingouin as pg


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


def quantile(data, q):
    """
    Calculate the q-th quantile of the data.

    Quantiles are points in a dataset that divide the data into intervals with
    equal probabilities. For example, the 0.5 quantile is the median, the 0.25
    quantile is the first quartile, and the 0.75 quantile is the third quartile.

    Parameters
    ----------
    data : list or array-like
        A list of numerical values. The input list must not be empty.
    q : float
        Quantile to compute, which must be between 0 and 1 inclusive.
        0 corresponds to the minimum, 0.5 corresponds to the median,
        and 1 corresponds to the maximum.

    Returns
    -------
    float
        The q-th quantile of the data.

    Raises
    ------
    ValueError
        If `data` is an empty list or if `q` is not between 0 and 1.

    Examples
    --------
    >>> quantile([1, 2, 3, 4, 5], 0.25)
    2.0
    >>> quantile([1, 2, 3, 4, 5], 0.5)
    3.0
    >>> quantile([1, 2, 3, 4, 5], 0.75)
    4.0

    Notes
    -----
    The quantile is calculated by first sorting the data, then finding
    the weighted interpolation between the two data points that correspond
    to the quantile index.

    If `q` is 0, the function returns the minimum value of the data.
    If `q` is 1, the function returns the maximum value of the data.

    This function supports interpolation when `q` does not exactly correspond
    to a data point index. The result will be a float value regardless of
    whether the input data contains integers.
    """
    if len(data) == 0:
        raise ValueError("Data cannot be empty")

    if not 0 <= q <= 1:
        raise ValueError("Quantile must be between 0 and 1")

    sorted_data = sorted(data)
    idx = (len(sorted_data) - 1) * q
    lower = int(idx)
    upper = lower + 1
    weight = idx - lower

    # If upper index is within bounds, interpolate between lower and upper
    if upper < len(sorted_data):
        result = sorted_data[lower] * (1 - weight) + sorted_data[upper] * weight
    else:
        result = sorted_data[lower]

    return float(result)  # Return as float to ensure consistent output type


def corrcoef(*args, **kwargs):
    r"""
    Pearson correlation coefficient and p-value for testing non-correlation.

    The Pearson correlation coefficient [1]_ measures the linear relationship
    between two datasets. Like other correlation
    coefficients, this one varies between -1 and +1 with 0 implying no
    correlation. Correlations of -1 or +1 imply an exact linear relationship.
    Positive correlations imply that as x increases, so does y. Negative
    correlations imply that as x increases, y decreases.

    This function also performs a test of the null hypothesis that the
    distributions underlying the samples are uncorrelated and normally
    distributed. (See Kowalski [3]_
    for a discussion of the effects of non-normality of the input on the
    distribution of the correlation coefficient.)
    The p-value roughly indicates the probability of an uncorrelated system
    producing datasets that have a Pearson correlation at least as extreme
    as the one computed from these datasets.

    Parameters
    ----------
    x : (N,) array_like
        Input array.
    y : (N,) array_like
        Input array.
    alternative : {'two-sided', 'greater', 'less'}, optional
        Defines the alternative hypothesis. Default is 'two-sided'.
        The following options are available:

        * 'two-sided': the correlation is nonzero
        * 'less': the correlation is negative (less than zero)
        * 'greater':  the correlation is positive (greater than zero)

        .. versionadded:: 1.9.0
    method : ResamplingMethod, optional
        Defines the method used to compute the p-value. If `method` is an
        instance of `PermutationMethod`/`MonteCarloMethod`, the p-value is
        computed using
        `scipy.stats.permutation_test`/`scipy.stats.monte_carlo_test` with the
        provided configuration options and other appropriate settings.
        Otherwise, the p-value is computed as documented in the notes.

        .. versionadded:: 1.11.0

    Returns
    -------
    result : `~scipy.stats._result_classes.PearsonRResult`
        An object with the following attributes:

        statistic : float
            Pearson product-moment correlation coefficient.
        pvalue : float
            The p-value associated with the chosen alternative.

        The object has the following method:

        confidence_interval(confidence_level, method)
            This computes the confidence interval of the correlation
            coefficient `statistic` for the given confidence level.
            The confidence interval is returned in a ``namedtuple`` with
            fields `low` and `high`. If `method` is not provided, the
            confidence interval is computed using the Fisher transformation
            [1]_. If `method` is an instance of `BootstrapMethod`, the
            confidence interval is computed using `scipy.stats.bootstrap` with
            the provided configuration options and other appropriate settings.
            In some cases, confidence limits may be NaN due to a degenerate
            resample, and this is typical for very small samples (~6
            observations).
    """

    def correlation_p_values(corr_matrix, n):
        p_matrix = np.zeros(corr_matrix.shape)

        for i in range(corr_matrix.shape[0]):
            for j in range(corr_matrix.shape[1]):
                if i != j:
                    r = corr_matrix[i, j]
                    t_stat = r * np.sqrt((n - 2) / (1 - r**2))
                    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=n-2))
                    p_matrix[i, j] = p_value
                else:
                    p_matrix[i, j] = 1  # No p-value for diagonal elements (correlation with self)
        
        return p_matrix

    R = np.corrcoef(*args, **kwargs, rowvar=False)
    P = correlation_p_values(R, len(args[0]))
    return (R, P)

def cov(*args, **kwargs):
    """
    Estimate a covariance matrix, given data and weights.

    Parameters
    ----------
    
        m : array_like
            A 1-D or 2-D array containing multiple variables and observations. Each row of `m` represents a
            variable, and each column a single observation of all those variables. Also see `rowvar` below.
        y : array_like, optional
            An additional set of variables and observations. `y` has the same form as that of `m`.
    
    Returns
    -------
    
        out : ndarray
            The covariance matrix of the variables. 
    """
    return np.cov(*args, **kwargs, rowvar=False)


def partialcorr(X, columns=None):
    """
    Wrapper for the pingouin partial_corr function to compute partial correlation matrix.

    Parameters
    ----------
    X : array-like or DataFrame
        A (n_samples, n_features) array or DataFrame with n_samples observations and n_features variables.
    columns : list of str, optional
        Column names for the DataFrame. If not provided, integers will be used as column names.

    Returns
    -------
    partial_corr_matrix : DataFrame
        Partial correlation matrix for the given variables.
    """
    if isinstance(X, pd.DataFrame):
        data = X
    else:
        if columns is None:
            columns = [f'var{i}' for i in range(X.shape[1])]
        data = pd.DataFrame(X, columns=columns)

    # Compute the partial correlation for each pair of variables using pingouin
    partial_corr_matrix = data.pcorr()

    return partial_corr_matrix


def covariance(data1, data2=None):
    """
    Calculate the covariance matrix for one or two sets of data.

    Parameters
    ----------
    data1 : array_like
        A 1D or 2D array representing the first set of data.
    data2 : array_like, optional
        A 1D array representing the second set of data. If provided, both `data1` and `data2`
        must have the same length.

    Returns
    -------
    cov_matrix : ndarray
        The covariance matrix.

    Raises
    ------
    ValueError
        If the input is empty or if `data1` and `data2` have mismatched lengths.
    """
    if data2 is None:
        data = np.asarray(data1)
        if data.size == 0:
            raise ValueError("Input data cannot be empty.")
        if data.ndim == 1:
            data = data[:, np.newaxis]
    else:
        data1 = np.asarray(data1)
        data2 = np.asarray(data2)
        if len(data1) != len(data2):
            raise ValueError("Input arrays must have the same length.")
        data = np.column_stack((data1, data2))

    return np.cov(data, rowvar=False)
