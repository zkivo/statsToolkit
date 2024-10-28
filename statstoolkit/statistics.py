import scipy.stats as stats 
import math
import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy.stats import t as t_dist
from numpy.linalg import inv, LinAlgError
import pingouin as pg
from statsmodels.stats.outliers_influence import summary_table
import statsmodels.formula.api as smf
from typing import Union, Optional, List
from statsmodels.formula.api import ols
from scipy.stats import kruskal
import matplotlib.pyplot as plt


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


def std(A, w=0, dim=None, vecdim=None, missingflag=None):
    """
    Compute standard deviation similar to MATLAB's std function with options for dimension, weighting, and missing values.

    Parameters
    ----------
    A : array-like
        Input array.
    w : int or array-like, optional
        Weighting scheme. Use 0 for sample standard deviation (default), 1 for population standard deviation,
        or an array of weights with the same length as the dimension of A.
    dim : int, optional
        Dimension along which to compute the standard deviation.
    vecdim : list of int, optional
        List of dimensions over which to compute the standard deviation.
    missingflag : str, optional
        Set to 'omitmissing' to ignore NaNs, or 'includemissing' to include NaNs in the calculation.

    Returns
    -------
    S : float or np.ndarray
        The standard deviation of the array.
    M : float or np.ndarray
        The mean of the array.
    """
    # Convert input to numpy array and handle missing values
    A = np.array(A, dtype=np.float64)

    if missingflag == "omitmissing":
        A = np.nan_to_num(A, nan=np.nanmean(A))  # Replace NaNs with mean of A
    elif missingflag == "includemissing" and np.isnan(A).any():
        return np.nan, np.nan  # Return NaN if missing values are to be included

    # Determine the dimensions to operate on
    if dim is not None:
        axis = dim
    elif vecdim is not None:
        axis = tuple(vecdim)
    else:
        # Default to the first non-singleton dimension
        axis = next((i for i in range(A.ndim) if A.shape[i] > 1), 0)

    # Compute mean and standard deviation depending on weight
    if isinstance(w, (int, float)) and w in [0, 1]:
        ddof = 1 if w == 0 else 0
        mean_A = np.mean(A, axis=axis, keepdims=True)

        if isinstance(axis, tuple):  # Handle vecdim scenario
            n_elements = np.prod([A.shape[i] for i in axis]) - ddof
            variance = np.sum((A - mean_A) ** 2, axis=axis) / n_elements
        else:
            n_elements = A.shape[axis] - ddof
            variance = np.sum((A - mean_A) ** 2, axis=axis) / n_elements

        S = np.sqrt(variance)
        M = np.mean(A, axis=axis)
    else:
        # Weighted standard deviation calculation
        mean_w = np.average(A, axis=axis, weights=w)
        variance_w = np.average((A - mean_w) ** 2, axis=axis, weights=w)
        S = np.sqrt(variance_w)
        M = mean_w

    # Squeeze dimensions to match MATLAB's behavior on dimensionality reduction
    if vecdim is not None or dim is not None:
        S = np.squeeze(S)
        M = np.squeeze(M)

    return S, M


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


def cov(data1, data2=None):
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


def fitlm(x, y):
    """
    Perform a simple linear regression of y on x, returning a dictionary with the results.

    Parameters
    ----------
    x : list or np.ndarray
        Predictor variable.
    y : list or np.ndarray
        Response variable.

    Returns
    -------
    dict
        Dictionary containing regression coefficients, statistics, and model diagnostics.

    Raises
    ------
    ValueError
        If there is no variance in both `x` and `y` or insufficient data for regression.
    """
    x, y = np.asarray(x), np.asarray(y)

    # Check for variance in x and y
    if len(x) < 2 or len(y) < 2:
        raise ValueError("Insufficient data for regression.")
    if np.var(x) == 0:
        raise ValueError("Zero variance in x.")
    if np.var(y) == 0:
        raise ValueError("Zero variance in y.")

    # Add constant for intercept in statsmodels
    x_with_const = sm.add_constant(x)
    model = sm.OLS(y, x_with_const).fit()

    # Prepare output dictionary
    summary = {
        "Coefficients": pd.DataFrame({
            "Estimate": model.params,
            "SE": model.bse,
            "tStat": model.tvalues,
            "pValue": model.pvalues
        }).rename(index={0: "Intercept"}),
        "Number of observations": int(model.nobs),
        "Error degrees of freedom": int(model.df_resid),
        "Root Mean Squared Error": np.sqrt(model.mse_resid),
        "R-squared": model.rsquared,
        "Adjusted R-squared": model.rsquared_adj,
        "F-statistic vs. constant model": model.fvalue,
        "p-value": model.f_pvalue
    }

    return summary


def regress(y, X, alpha=0.05):
    """
    Perform multiple linear regression similar to MATLAB's regress function.

    Parameters:
    y : array-like
        Response variable.
    X : array-like
        Predictor matrix including an intercept column.
    alpha : float, optional
        Significance level for confidence intervals. Default is 0.05.

    Returns:
    b : array
        Coefficients of the linear regression.
    bint : array
        Confidence intervals for the coefficients.
    r : array
        Residuals of the regression.
    rint : array
        Intervals for residuals (outlier diagnosis).
    stats : dict
        Dictionary containing R-squared, F-statistic, p-value, and error variance.
    """
    # Ensure the predictor matrix X includes an intercept column (all ones)
    if not np.all(X[:, 0] == 1):
        raise ValueError("The first column of X must be ones for intercept.")

    # Remove rows with NaN values in y or X
    mask = ~np.isnan(y) & ~np.isnan(X).any(axis=1)
    y_clean = y[mask]
    X_clean = X[mask, :]

    # Check for insufficient data
    if X_clean.shape[0] <= X_clean.shape[1]:
        raise ValueError("Insufficient data points for regression analysis.")

    # Check for variance in predictors
    if np.all(np.var(X_clean[:, 1:], axis=0) == 0):  # Ignore intercept column
        raise ValueError("Predictor matrix X has no variance in some columns.")

    # Check for variance in response
    if np.var(y_clean) == 0:
        raise ValueError("Response variable y has no variance.")

    # Fit the model
    model = sm.OLS(y_clean, X_clean).fit()

    # Extract coefficients and confidence intervals
    b = model.params
    bint = model.conf_int(alpha=alpha)

    # Compute residuals
    r = model.resid
    st, data, ss2 = summary_table(model, alpha=alpha)
    rint = data[:, [4, 5]]  # The lower and upper bounds for residuals

    # Compute additional statistics
    stats = {
        "R-squared": model.rsquared,
        "F-statistic": model.fvalue,
        "p-value": model.f_pvalue,
        "Error variance": model.mse_resid
    }

    # Calculate the residual standard error (RSE)
    degrees_of_freedom = model.df_resid
    residual_standard_error = np.sqrt(np.sum(r**2) / degrees_of_freedom)

    # Set confidence level, e.g., 95%
    t_value = t_dist.ppf(1 - alpha/2, degrees_of_freedom)

    # Calculate the confidence interval for each residual
    lower_bound = np.array(r - t_value * residual_standard_error)
    upper_bound = np.array(r + t_value * residual_standard_error)

    rint = np.column_stack((lower_bound, upper_bound))

    return b, bint, r, rint, stats


def ttest(x, y=None, m=0, alpha=0.05, alternative='two-sided'):
    """
    Perform a one-sample or paired-sample t-test.

    Parameters:
    x : array-like
        The first sample data.
    y : array-like, optional
        The second sample data for paired t-test. Default is None.
    m : float, optional
        The hypothesized population mean for one-sample t-test. Default is 0.
    alpha : float, optional
        Significance level for confidence intervals. Default is 0.05.
    alternative : {'two-sided', 'greater', 'less'}, optional
        Specifies the alternative hypothesis. Default is 'two-sided'.

    Returns:
    h : int
        Test decision: 1 if the null hypothesis is rejected, 0 otherwise.
    p : float
        p-value of the test.
    ci : tuple
        Confidence interval for the mean difference.
    stats : dict
        Contains the t-statistic and degrees of freedom (df).
    """

    # Convert inputs to numpy arrays
    x = np.asarray(x)
    if y is not None:
        y = np.asarray(y)

    # Validate inputs
    if not (0 < alpha < 1):
        raise ValueError("Alpha must be between 0 and 1.")
    if alternative not in {'two-sided', 'greater', 'less'}:
        raise ValueError("Alternative hypothesis must be 'two-sided', 'greater', or 'less'.")
    if len(x) == 0 or (y is not None and len(y) == 0):
        raise ValueError("Input samples must not be empty.")

    # Perform test based on the input parameters
    if y is None:
        # One-sample t-test
        t_stat, p_val = stats.ttest_1samp(x, m)
    else:
        # Paired-sample t-test
        if len(x) != len(y):
            raise ValueError("Both samples must have the same length for a paired t-test.")
        t_stat, p_val = stats.ttest_rel(x, y)

    # Adjust p-value for one-sided tests
    if alternative == 'greater':
        if t_stat < 0:
            p_val = 1.0  # Fail the test since we're looking for t_stat > 0
        else:
            p_val /= 2
    elif alternative == 'less':
        if t_stat > 0:
            p_val = 1.0  # Fail the test since we're looking for t_stat < 0
        else:
            p_val /= 2

    # Compute the test decision
    h = int(p_val < alpha)

    # Confidence interval calculation
    mean_diff = np.mean(x) - (np.mean(y) if y is not None else 0)
    se = stats.sem(x - y if y is not None else x)
    df = len(x) - 1
    t_crit = stats.t.ppf(1 - alpha / 2, df=df)
    ci = (mean_diff - t_crit * se, mean_diff + t_crit * se)
    sd = np.std(x, ddof=1) if y is None else np.std(x - y, ddof=1)

    # Collect statistics
    stats_dict = {
        "t_stat": t_stat,
        "df": df,
        "sd": sd
    }

    return h, p_val, ci, stats_dict


def ttest2(x, y, alpha=0.05, equal_var=True, alternative='two-sided'):
    """
    Perform two-sample t-test on two independent samples.

    Parameters:
    x, y : array-like
        The two independent samples.
    alpha : float, optional
        Significance level for the test. Must be between 0 and 1. Default is 0.05.
    equal_var : bool, optional
        If True (default), perform a standard independent 2-sample test that assumes equal population variances.
        If False, perform Welch's t-test, which does not assume equal population variance.
    alternative : {'two-sided', 'greater', 'less'}, optional
        Defines the alternative hypothesis.

    Returns:
    h : int
        Test decision: 1 if the null hypothesis is rejected at the specified significance level, 0 otherwise.
    p : float
        p-value of the test.
    ci : tuple
        Confidence interval for the mean difference.
    stats : dict
        Dictionary containing test statistic ('t_stat') and degrees of freedom ('df').
    """
    from scipy.stats import ttest_ind

    # Input validation
    if len(x) == 0 or len(y) == 0:
        raise ValueError("Input samples must not be empty.")
    if not (0 < alpha < 1):
        raise ValueError("Alpha must be between 0 and 1.")
    if alternative not in {'two-sided', 'greater', 'less'}:
        raise ValueError("Invalid alternative hypothesis specified.")

    # Perform the t-test
    t_stat, p = ttest_ind(x, y, equal_var=equal_var, alternative=alternative)

    # Confidence interval
    mean_diff = np.mean(x) - np.mean(y)
    se_diff = np.sqrt((np.var(x, ddof=1) / len(x)) + (np.var(y, ddof=1) / len(y)))
    df = len(x) + len(y) - 2
    t_critical = abs(t_dist.ppf(alpha / 2, df=df))
    ci = (mean_diff - t_critical * se_diff, mean_diff + t_critical * se_diff)

    # Test decision
    h = int(p < alpha)

    # Statistics output
    stats = {
        "t_stat": t_stat,
        "df": float(df)
    }

    return h, p, ci, stats


class AnovaResult:
    def __init__(self, anova_table):
        self.anova_table = anova_table

    def summary(self):
        """Return the ANOVA table as a formatted summary."""
        return self.anova_table


def anova(y=None, factors=None, data=None, formula=None, response=None, sum_of_squares='type I'):
    """
    Perform Analysis of Variance (ANOVA) with support for one-way, two-way, and N-way models.

    Parameters:
    - y: array-like, optional
        Response variable for one-way ANOVA or with factors for multi-way ANOVA.
    - factors: list of array-like, optional
        A list of factors for multi-way ANOVA.
    - data: DataFrame, optional
        DataFrame containing data for formula-based or table-based ANOVA.
    - formula: str, optional
        Formula in Wilkinson notation for ANOVA (e.g., 'y ~ A + B + A:B').
    - response: str, optional
        Name of the response variable in data if using table-based ANOVA.
    - sum_of_squares: str, optional
        Type of sum of squares for ANOVA calculations. Options are 'type I', 'type II', or 'type III'.

    Returns:
    - anova_table: DataFrame
        ANOVA table with sum of squares, degrees of freedom, F-statistic, and p-value.

    Raises:
    - ValueError: If input parameters are invalid or if any factor has only one level.
    """

    # Ensure valid sum of squares type
    valid_sstypes = {'type I': 1, 'type II': 2, 'type III': 3}
    if sum_of_squares not in valid_sstypes:
        raise ValueError("sum_of_squares must be 'type I', 'type II', or 'type III'.")

    # Case 1: Formula-based ANOVA
    if formula:
        if data is None:
            raise ValueError("Data must be provided with a formula.")
        model = ols(formula, data=data).fit()

    # Case 2: Multi-way ANOVA with factors and response variable
    elif factors is not None:
        if y is None:
            raise ValueError("Response data 'y' must be provided with factors.")

        # Check if any factor has only one level
        for i, factor in enumerate(factors):
            if len(set(factor)) < 2:
                raise ValueError(f"Factor 'factor_{i}' must have at least two levels.")

        # Prepare DataFrame and formula
        factor_data = {f"factor_{i}": pd.Categorical(factor) for i, factor in enumerate(factors)}
        df = pd.DataFrame(factor_data)
        df['y'] = y
        formula_parts = ' + '.join(f"C(factor_{i})" for i in range(len(factors)))
        formula = f"y ~ {formula_parts}"
        model = ols(formula, data=df).fit()

    # Case 3: One-way ANOVA with response matrix (2D y)
    elif isinstance(y, np.ndarray) and y.ndim == 2:
        if y.shape[1] < 2:
            raise ValueError("For one-way ANOVA, 'y' must have at least two columns.")

        # Reshape y into a long format for one-way ANOVA
        df = pd.DataFrame(y, columns=[f"factor_{i}" for i in range(y.shape[1])])
        df = pd.melt(df, var_name="factor", value_name="y")
        model = ols("y ~ C(factor)", data=df).fit()

    # Case 4: Table-based ANOVA with response variable name
    elif response and data is not None:
        if response not in data.columns:
            raise ValueError("Response variable not found in data.")
        formula = f"{response} ~ " + " + ".join([f"C({col})" for col in data.columns if col != response])
        model = ols(formula, data=data).fit()

    else:
        raise ValueError(
            "Invalid input parameters. Provide `formula` with `data`, `factors` with `y`, or a 2D `y` for one-way ANOVA.")

    # Compute ANOVA table
    anova_table = sm.stats.anova_lm(model, typ=valid_sstypes[sum_of_squares])

    return anova_table


def kruskalwallis(x, group=None, displayopt=False):
    """
    Perform the Kruskal-Wallis H-test for independent samples.

    Parameters:
    x : array-like
        Data values. If x is a 2D array, each column is treated as a separate group.
    group : array-like, optional
        Group labels for the 1D array x. If provided, `x` must be 1D.
    displayopt : bool, optional
        If True, display the ANOVA table and boxplot. Default is False.

    Returns:
    p : float
        P-value for the test.
    tbl : DataFrame
        ANOVA table containing source, H-statistic, and p-value.
    stats : dict
        Dictionary containing test statistics, including test_statistic, p_value, and df.
    """
    # Group handling: convert input to a list of groups
    if isinstance(x, np.ndarray) and x.ndim == 2:
        groups = [x[:, i] for i in range(x.shape[1])]
    elif group is not None:
        if len(x) != len(group):
            raise ValueError("x and group must have the same length")
        # Group data by unique group labels
        unique_groups = np.unique(group)
        groups = [np.array([x[i] for i in range(len(x)) if group[i] == g]) for g in unique_groups]
    else:
        raise ValueError("Either provide a 2D array for x or a group label array for 1D x.")

    # Check for identical values in all groups
    if all(np.all(group == group[0]) for group in groups):
        raise ValueError("All values in each group are identical; Kruskal-Wallis test cannot be performed.")

    # Run the Kruskal-Wallis test
    h_stat, p_value = stats.kruskal(*groups)

    # Create ANOVA table as DataFrame
    tbl = pd.DataFrame({
        "Source": ["Kruskal-Wallis", "Error"],
        "H": [h_stat, np.nan],
        "p-value": [p_value, np.nan]
    })

    # Prepare stats dictionary
    stats_dict = {
        "test_statistic": h_stat,
        "p_value": p_value,
        "df": len(groups) - 1,
        "groups": len(groups)
    }

    # Display results if requested
    if displayopt:
        print(tbl)
        plt.boxplot(groups, labels=unique_groups if group is not None else range(len(groups)))
        plt.title("Kruskal-Wallis Test")
        plt.show()

    return float(p_value), tbl, stats_dict
