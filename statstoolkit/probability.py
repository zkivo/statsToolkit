from scipy import stats as st

def binompdf(*args, **kwargs):
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