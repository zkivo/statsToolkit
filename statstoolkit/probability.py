from scipy import stats as st


def binopdf(*args, **kwargs):
    """Probability mass function at k of the given RV.

    Parameters
    ----------
    k : array_like
        Quantiles.
    arg1, arg2, arg3,... : array_like
        The shape parameter(s) for the distribution (see docstring of the
        instance object for more information)
    loc : array_like, optional
        Location parameter (default=0).

    Returns
    -------
    pmf : array_like
        Probability mass function evaluated at k

    """
    return st.binom.pmf(*args, **kwargs)


def poisspdf(*args, **kwargs):
    """Probability mass function at k of the given RV.

    Parameters
    ----------
    k : array_like
        Quantiles.
    arg1, arg2, arg3,... : array_like
        The shape parameter(s) for the distribution (see docstring of the
        instance object for more information)
    loc : array_like, optional
        Location parameter (default=0).

    Returns
    -------
    pmf : array_like
        Probability mass function evaluated at k

    """
    return st.poisson.pmf(*args, **kwargs)


def geopdf(*args, **kwargs):
    """Probability mass function at k of the given RV.

    Parameters
    ----------
    k : array_like
        Quantiles.
    arg1, arg2, arg3,... : array_like
        The shape parameter(s) for the distribution (see docstring of the
        instance object for more information)
    loc : array_like, optional
        Location parameter (default=0).

    Returns
    -------
    pmf : array_like
        Probability mass function evaluated at k

    """
    return st.geom.pmf(*args, **kwargs)


def nbinpdf(*args, **kwargs):
    """Probability mass function at k of the given RV.

    Parameters
    ----------
    k : array_like
        Quantiles.
    arg1, arg2, arg3,... : array_like
        The shape parameter(s) for the distribution (see docstring of the
        instance object for more information)
    loc : array_like, optional
        Location parameter (default=0).

    Returns
    -------
    pmf : array_like
        Probability mass function evaluated at k

    """
    return st.nbinom.pmf(*args, **kwargs)


def hygepdf(*args, **kwargs):
    """Probability mass function at k of the given RV.

    Parameters
    ----------
    k : array_like
        Quantiles.
    arg1, arg2, arg3,... : array_like
        The shape parameter(s) for the distribution (see docstring of the
        instance object for more information)
    loc : array_like, optional
        Location parameter (default=0).

    Returns
    -------
    pmf : array_like
        Probability mass function evaluated at k

    """
    return st.hypergeom.pmf(*args, **kwargs)


def betapdf(*args, **kwargs):
    """Probability density function at x of the given RV.

    Parameters
    ----------
    x : array_like
        quantiles
    arg1, arg2, arg3,... : array_like
        The shape parameter(s) for the distribution (see docstring of the
        instance object for more information)
    loc : array_like, optional
        location parameter (default=0)
    scale : array_like, optional
        scale parameter (default=1)

    Returns
    -------
    pdf : ndarray
        Probability density function evaluated at x

    """
    return st.beta.pdf(*args, **kwargs)


def chi2pdf(*args, **kwargs):
    """Probability density function at x of the given RV.

    Parameters
    ----------
    x : array_like
        quantiles
    arg1, arg2, arg3,... : array_like
        The shape parameter(s) for the distribution (see docstring of the
        instance object for more information)
    loc : array_like, optional
        location parameter (default=0)
    scale : array_like, optional
        scale parameter (default=1)

    Returns
    -------
    pdf : ndarray
        Probability density function evaluated at x

    """
    return st.chi2.pdf(*args, **kwargs)


def exppdf(*args, **kwargs):
    """Probability density function at x of the given RV.

    Parameters
    ----------
    x : array_like
        quantiles
    arg1, arg2, arg3,... : array_like
        The shape parameter(s) for the distribution (see docstring of the
        instance object for more information)
    loc : array_like, optional
        location parameter (default=0)
    scale : array_like, optional
        scale parameter (default=1)

    Returns
    -------
    pdf : ndarray
        Probability density function evaluated at x

    """
    return st.expon.pdf(*args, **kwargs)


def fpdf(*args, **kwargs):
    """Probability density function at x of the given RV.

    Parameters
    ----------
    x : array_like
        quantiles
    arg1, arg2, arg3,... : array_like
        The shape parameter(s) for the distribution (see docstring of the
        instance object for more information)
    loc : array_like, optional
        location parameter (default=0)
    scale : array_like, optional
        scale parameter (default=1)

    Returns
    -------
    pdf : ndarray
        Probability density function evaluated at x

    """
    return st.f.pdf(*args, **kwargs)


def normpdf(*args, **kwargs):
    """Probability density function at x of the given RV.

    Parameters
    ----------
    x : array_like
        quantiles
    arg1, arg2, arg3,... : array_like
        The shape parameter(s) for the distribution (see docstring of the
        instance object for more information)
    loc : array_like, optional
        location parameter (default=0)
    scale : array_like, optional
        scale parameter (default=1)

    Returns
    -------
    pdf : ndarray
        Probability density function evaluated at x

    """
    return st.norm.pdf(*args, **kwargs)


def lognpdf(*args, **kwargs):
    """Probability density function at x of the given RV.

    Parameters
    ----------
    x : array_like
        quantiles
    arg1, arg2, arg3,... : array_like
        The shape parameter(s) for the distribution (see docstring of the
        instance object for more information)
    loc : array_like, optional
        location parameter (default=0)
    scale : array_like, optional
        scale parameter (default=1)

    Returns
    -------
    pdf : ndarray
        Probability density function evaluated at x

    """
    return st.lognorm.pdf(*args, **kwargs)


def tpdf(*args, **kwargs):
    """Probability density function at x of the given RV.

    Parameters
    ----------
    x : array_like
        quantiles
    arg1, arg2, arg3,... : array_like
        The shape parameter(s) for the distribution (see docstring of the
        instance object for more information)
    loc : array_like, optional
        location parameter (default=0)
    scale : array_like, optional
        scale parameter (default=1)

    Returns
    -------
    pdf : ndarray
        Probability density function evaluated at x

    """
    return st.t.pdf(*args, **kwargs)


def wblpdf(*args, **kwargs):
    """Probability density function at x of the given RV.

    Parameters
    ----------
    x : array_like
        quantiles
    arg1, arg2, arg3,... : array_like
        The shape parameter(s) for the distribution (see docstring of the
        instance object for more information)
    loc : array_like, optional
        location parameter (default=0)
    scale : array_like, optional
        scale parameter (default=1)

    Returns
    -------
    pdf : ndarray
        Probability density function evaluated at x

    """
    return st.weibull_min.pdf(*args, **kwargs)


def mvnpdf(*args, **kwargs):
    """Probability density function at x of the given RV.

    Parameters
    ----------
    x : array_like
        quantiles
    arg1, arg2, arg3,... : array_like
        The shape parameter(s) for the distribution (see docstring of the
        instance object for more information)
    loc : array_like, optional
        location parameter (default=0)
    scale : array_like, optional
        scale parameter (default=1)

    Returns
    -------
    pdf : ndarray
        Probability density function evaluated at x

    """
    return st.multivariate_normal.pdf(*args, **kwargs)

def binocdf(*args, **kwargs):
    """Cumulative distribution function at k of the given RV.

    Parameters
    ----------
    k : array_like
        Quantiles.
    arg1, arg2, arg3,... : array_like
        The shape parameter(s) for the distribution (see docstring of the
        instance object for more information)
    loc : array_like, optional
        Location parameter (default=0).

    Returns
    -------
    cdf : array_like
        Cumulative distribution function evaluated at k

    """
    return st.binom.cdf(*args, **kwargs)

def poisscdf(*args, **kwargs):
    """Cumulative distribution function at k of the given RV.

    Parameters
    ----------
    k : array_like
        Quantiles.
    arg1, arg2, arg3,... : array_like
        The shape parameter(s) for the distribution (see docstring of the
        instance object for more information)
    loc : array_like, optional
        Location parameter (default=0).

    Returns
    -------
    cdf : array_like
        Cumulative distribution function evaluated at k

    """
    return st.poisson.cdf(*args, **kwargs)

def geocdf(*args, **kwargs):
    """Cumulative distribution function at k of the given RV.

    Parameters
    ----------
    k : array_like
        Quantiles.
    arg1, arg2, arg3,... : array_like
        The shape parameter(s) for the distribution (see docstring of the
        instance object for more information)
    loc : array_like, optional
        Location parameter (default=0).

    Returns
    -------
    cdf : array_like
        Cumulative distribution function evaluated at k

    """
    return st.geom.cdf(*args, **kwargs)

def nbincdf(*args, **kwargs):
    """Cumulative distribution function at k of the given RV.

    Parameters
    ----------
    k : array_like
        Quantiles.
    arg1, arg2, arg3,... : array_like
        The shape parameter(s) for the distribution (see docstring of the
        instance object for more information)
    loc : array_like, optional
        Location parameter (default=0).

    Returns
    -------
    cdf : array_like
        Cumulative distribution function evaluated at k

    """
    return st.nbinom.cdf(*args, **kwargs)

def hygecdf(*args, **kwargs):
    """Cumulative distribution function at k of the given RV.

    Parameters
    ----------
    k : array_like
        Quantiles.
    arg1, arg2, arg3,... : array_like
        The shape parameter(s) for the distribution (see docstring of the
        instance object for more information)
    loc : array_like, optional
        Location parameter (default=0).

    Returns
    -------
    cdf : array_like
        Cumulative distribution function evaluated at k

    """
    return st.hypergeom.cdf(*args, **kwargs)

def betacdf(*args, **kwargs):
    """Cumulative distribution function at x of the given RV.

    Parameters
    ----------
    x : array_like
        quantiles
    arg1, arg2, arg3,... : array_like
        The shape parameter(s) for the distribution (see docstring of the
        instance object for more information)
    loc : array_like, optional
        location parameter (default=0)
    scale : array_like, optional
        scale parameter (default=1)

    Returns
    -------
    cdf : ndarray
        Cumulative distribution function evaluated at x

    """
    return st.beta.cdf(*args, **kwargs)

def chi2cdf(*args, **kwargs):
    """Cumulative distribution function at x of the given RV.

    Parameters
    ----------
    x : array_like
        quantiles
    arg1, arg2, arg3,... : array_like
        The shape parameter(s) for the distribution (see docstring of the
        instance object for more information)
    loc : array_like, optional
        location parameter (default=0)
    scale : array_like, optional
        scale parameter (default=1)

    Returns
    -------
    cdf : ndarray
        Cumulative distribution function evaluated at x

    """
    return st.chi2.cdf(*args, **kwargs)

def expcdf(*args, **kwargs):
    """Cumulative distribution function at x of the given RV.

    Parameters
    ----------
    x : array_like
        quantiles
    arg1, arg2, arg3,... : array_like
        The shape parameter(s) for the distribution (see docstring of the
        instance object for more information)
    loc : array_like, optional
        location parameter (default=0)
    scale : array_like, optional
        scale parameter (default=1)

    Returns
    -------
    cdf : ndarray
        Cumulative distribution function evaluated at x

    """
    return st.expon.cdf(*args, **kwargs)

def fcdf(*args, **kwargs):
    """Cumulative distribution function at x of the given RV.

    Parameters
    ----------
    x : array_like
        quantiles
    arg1, arg2, arg3,... : array_like
        The shape parameter(s) for the distribution (see docstring of the
        instance object for more information)
    loc : array_like, optional
        location parameter (default=0)
    scale : array_like, optional
        scale parameter (default=1)

    Returns
    -------
    cdf : ndarray
        Cumulative distribution function evaluated at x

    """
    return st.f.cdf(*args, **kwargs)

def normcdf(*args, **kwargs):
    """Cumulative distribution function at x of the given RV.

    Parameters
    ----------
    x : array_like
        quantiles
    arg1, arg2, arg3,... : array_like
        The shape parameter(s) for the distribution (see docstring of the
        instance object for more information)
    loc : array_like, optional
        location parameter (default=0)
    scale : array_like, optional
        scale parameter (default=1)

    Returns
    -------
    cdf : ndarray
        Cumulative distribution function evaluated at x

    """
    return st.norm.cdf(*args, **kwargs)

def logncdf(*args, **kwargs):
    """Cumulative distribution function at x of the given RV.

    Parameters
    ----------
    x : array_like
        quantiles
    arg1, arg2, arg3,... : array_like
        The shape parameter(s) for the distribution (see docstring of the
        instance object for more information)
    loc : array_like, optional
        location parameter (default=0)
    scale : array_like, optional
        scale parameter (default=1)

    Returns
    -------
    cdf : ndarray
        Cumulative distribution function evaluated at x

    """
    return st.lognorm.cdf(*args, **kwargs)

def tcdf(*args, **kwargs):
    """Cumulative distribution function at x of the given RV.

    Parameters
    ----------
    x : array_like
        quantiles
    arg1, arg2, arg3,... : array_like
        The shape parameter(s) for the distribution (see docstring of the
        instance object for more information)
    loc : array_like, optional
        location parameter (default=0)
    scale : array_like, optional
        scale parameter (default=1)

    Returns
    -------
    cdf : ndarray
        Cumulative distribution function evaluated at x

    """
    return st.t.cdf(*args, **kwargs)

def wblcdf(*args, **kwargs):
    """Cumulative distribution function at x of the given RV.

    Parameters
    ----------
    x : array_like
        quantiles
    arg1, arg2, arg3,... : array_like
        The shape parameter(s) for the distribution (see docstring of the
        instance object for more information)
    loc : array_like, optional
        location parameter (default=0)
    scale : array_like, optional
        scale parameter (default=1)

    Returns
    -------
    cdf : ndarray
        Cumulative distribution function evaluated at x

    """
    return st.weibull_min.cdf(*args, **kwargs)

def norminv(*args, **kwargs):
    """Percent point function (inverse of `cdf`) at q of the given RV.

    Parameters
    ----------
    q : array_like
        lower tail probability
    arg1, arg2, arg3,... : array_like
        The shape parameter(s) for the distribution (see docstring of the
        instance object for more information)
    loc : array_like, optional
        location parameter (default=0)
    scale : array_like, optional
        scale parameter (default=1)

    Returns
    -------
    x : array_like
        quantile corresponding to the lower tail probability q.

    """
    return st.norm.ppf(*args, **kwargs)

def tinv(*args, **kwargs):
    """Percent point function (inverse of `cdf`) at q of the given RV.

    Parameters
    ----------
    q : array_like
        lower tail probability
    arg1, arg2, arg3,... : array_like
        The shape parameter(s) for the distribution (see docstring of the
        instance object for more information)
    loc : array_like, optional
        location parameter (default=0)
    scale : array_like, optional
        scale parameter (default=1)

    Returns
    -------
    x : array_like
        quantile corresponding to the lower tail probability q.

    """
    return st.t.ppf(*args, **kwargs)

def chi2inv(*args, **kwargs):
    """Percent point function (inverse of `cdf`) at q of the given RV.

    Parameters
    ----------
    q : array_like
        lower tail probability
    arg1, arg2, arg3,... : array_like
        The shape parameter(s) for the distribution (see docstring of the
        instance object for more information)
    loc : array_like, optional
        location parameter (default=0)
    scale : array_like, optional
        scale parameter (default=1)

    Returns
    -------
    x : array_like
        quantile corresponding to the lower tail probability q.

    """
    return st.chi2.ppf(*args, **kwargs)

def finv(*args, **kwargs):
    """Percent point function (inverse of `cdf`) at q of the given RV.

    Parameters
    ----------
    q : array_like
        lower tail probability
    arg1, arg2, arg3,... : array_like
        The shape parameter(s) for the distribution (see docstring of the
        instance object for more information)
    loc : array_like, optional
        location parameter (default=0)
    scale : array_like, optional
        scale parameter (default=1)

    Returns
    -------
    x : array_like
        quantile corresponding to the lower tail probability q.

    """
    return st.f.ppf(*args, **kwargs)
