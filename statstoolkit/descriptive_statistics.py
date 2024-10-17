import scipy.stats as stats 
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
    return stats.pearsonr(*args, **kwargs)
