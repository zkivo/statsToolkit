from scipy.integrate import quad, dblquad
import numpy as np
import pandas as pd
import re


def readmatrix(filename, range_str=None):
    """
    Read a specific range of cells or columns from an Excel file into a matrix (DataFrame).

    Parameters
    ----------
    filename : str
        The name of the Excel file (e.g., 'name.xlsx').
    range_str : str, optional
        The cell range in Excel notation (e.g., 'A1:E5') or column range (e.g., 'A:E').

    Returns
    -------
    DataFrame
        A pandas DataFrame containing the data from the specified range.

    Examples
    --------
    >>> readmatrix('data.xlsx', 'A1:E5')
    >>> readmatrix('data.xlsx', 'A:E')
    """
    if range_str is None:
        return pd.read_excel(filename, header=None)

    # Check if the range_str is a full cell range (e.g., A1:E5)
    match = re.match(r'([A-Z]+)(\d+):([A-Z]+)(\d+)', range_str)

    if match:
        start_col, start_row, end_col, end_row = match.groups()
        cols = f"{start_col}:{end_col}"
        data = pd.read_excel(filename, usecols=cols, header=None)
        start_row_idx = int(start_row) - 1
        end_row_idx = int(end_row)
        return data.iloc[start_row_idx:end_row_idx]

    # Check if the range_str is a column-only range (e.g., A:E)
    match = re.match(r'([A-Z]+):([A-Z]+)', range_str)

    if match:
        start_col, end_col = match.groups()
        cols = f"{start_col}:{end_col}"
        return pd.read_excel(filename, usecols=cols, header=None)

    raise ValueError(f"Invalid range format: {range_str}")


def linspace(*args, **kwargs):
    """
    Return evenly spaced numbers over a specified interval.

    Returns `num` evenly spaced samples, calculated over the
    interval [`start`, `stop`].

    The endpoint of the interval can optionally be excluded.

    .. versionchanged:: 1.16.0
        Non-scalar `start` and `stop` are now supported.

    .. versionchanged:: 1.20.0
        Values are rounded towards ``-inf`` instead of ``0`` when an
        integer ``dtype`` is specified. The old behavior can
        still be obtained with ``np.linspace(start, stop, num).astype(int)``

    Parameters
    ----------
    start : array_like
        The starting value of the sequence.
    stop : array_like
        The end value of the sequence, unless `endpoint` is set to False.
        In that case, the sequence consists of all but the last of ``num + 1``
        evenly spaced samples, so that `stop` is excluded.  Note that the step
        size changes when `endpoint` is False.
    num : int, optional
        Number of samples to generate. Default is 50. Must be non-negative.
    endpoint : bool, optional
        If True, `stop` is the last sample. Otherwise, it is not included.
        Default is True.
    retstep : bool, optional
        If True, return (`samples`, `step`), where `step` is the spacing
        between samples.
    dtype : dtype, optional
        The type of the output array.  If `dtype` is not given, the data type
        is inferred from `start` and `stop`. The inferred dtype will never be
        an integer; `float` is chosen even if the arguments would produce an
        array of integers.

        .. versionadded:: 1.9.0

    axis : int, optional
        The axis in the result to store the samples.  Relevant only if start
        or stop are array-like.  By default (0), the samples will be along a
        new axis inserted at the beginning. Use -1 to get an axis at the end.

        .. versionadded:: 1.16.0

    Returns
    -------
    samples : ndarray
        There are `num` equally spaced samples in the closed interval
        ``[start, stop]`` or the half-open interval ``[start, stop)``
        (depending on whether `endpoint` is True or False).
    step : float, optional
        Only returned if `retstep` is True

        Size of spacing between samples.


    See Also
    --------
    arange : Similar to `linspace`, but uses a step size (instead of the
             number of samples).
    geomspace : Similar to `linspace`, but with numbers spaced evenly on a log
                scale (a geometric progression).
    logspace : Similar to `geomspace`, but with the end points specified as
               logarithms.
    :ref:`how-to-partition`

    Examples
    --------
    >>> np.linspace(2.0, 3.0, num=5)
    array([2.  , 2.25, 2.5 , 2.75, 3.  ])
    >>> np.linspace(2.0, 3.0, num=5, endpoint=False)
    array([2. ,  2.2,  2.4,  2.6,  2.8])
    >>> np.linspace(2.0, 3.0, num=5, retstep=True)
    (array([2.  ,  2.25,  2.5 ,  2.75,  3.  ]), 0.25)

    Graphical illustration:

    >>> import matplotlib.pyplot as plt
    >>> N = 8
    >>> y = np.zeros(N)
    >>> x1 = np.linspace(0, 10, N, endpoint=True)
    >>> x2 = np.linspace(0, 10, N, endpoint=False)
    >>> plt.plot(x1, y, 'o')
    [<matplotlib.lines.Line2D object at 0x...>]
    >>> plt.plot(x2, y + 0.5, 'o')
    [<matplotlib.lines.Line2D object at 0x...>]
    >>> plt.ylim([-0.5, 1])
    (-0.5, 1)
    >>> plt.show()

    """
    return np.linspace(*args, **kwargs)


def meshgrid(*args, **kwargs):
    """
    Return a list of coordinate matrices from coordinate vectors.

    Make N-D coordinate arrays for vectorized evaluations of
    N-D scalar/vector fields over N-D grids, given
    one-dimensional coordinate arrays x1, x2,..., xn.

    .. versionchanged:: 1.9
       1-D and 0-D cases are allowed.

    Parameters
    ----------
    x1, x2,..., xn : array_like
        1-D arrays representing the coordinates of a grid.
    indexing : {'xy', 'ij'}, optional
        Cartesian ('xy', default) or matrix ('ij') indexing of output.
        See Notes for more details.

        .. versionadded:: 1.7.0
    sparse : bool, optional
        If True the shape of the returned coordinate array for dimension *i*
        is reduced from ``(N1, ..., Ni, ... Nn)`` to
        ``(1, ..., 1, Ni, 1, ..., 1)``.  These sparse coordinate grids are
        intended to be use with :ref:`basics.broadcasting`.  When all
        coordinates are used in an expression, broadcasting still leads to a
        fully-dimensonal result array.

        Default is False.

        .. versionadded:: 1.7.0
    copy : bool, optional
        If False, a view into the original arrays are returned in order to
        conserve memory.  Default is True.  Please note that
        ``sparse=False, copy=False`` will likely return non-contiguous
        arrays.  Furthermore, more than one element of a broadcast array
        may refer to a single memory location.  If you need to write to the
        arrays, make copies first.

        .. versionadded:: 1.7.0

    Returns
    -------
    X1, X2,..., XN : list of ndarrays
        For vectors `x1`, `x2`,..., `xn` with lengths ``Ni=len(xi)``,
        returns ``(N1, N2, N3,..., Nn)`` shaped arrays if indexing='ij'
        or ``(N2, N1, N3,..., Nn)`` shaped arrays if indexing='xy'
        with the elements of `xi` repeated to fill the matrix along
        the first dimension for `x1`, the second for `x2` and so on.

    Notes
    -----
    This function supports both indexing conventions through the indexing
    keyword argument.  Giving the string 'ij' returns a meshgrid with
    matrix indexing, while 'xy' returns a meshgrid with Cartesian indexing.
    In the 2-D case with inputs of length M and N, the outputs are of shape
    (N, M) for 'xy' indexing and (M, N) for 'ij' indexing.  In the 3-D case
    with inputs of length M, N and P, outputs are of shape (N, M, P) for
    'xy' indexing and (M, N, P) for 'ij' indexing.  The difference is
    illustrated by the following code snippet::

        xv, yv = np.meshgrid(x, y, indexing='ij')
        for i in range(nx):
            for j in range(ny):
                # treat xv[i,j], yv[i,j]

        xv, yv = np.meshgrid(x, y, indexing='xy')
        for i in range(nx):
            for j in range(ny):
                # treat xv[j,i], yv[j,i]

    In the 1-D and 0-D case, the indexing and sparse keywords have no effect.

    See Also
    --------
    mgrid : Construct a multi-dimensional "meshgrid" using indexing notation.
    ogrid : Construct an open multi-dimensional "meshgrid" using indexing
            notation.
    how-to-index

    Examples
    --------
    >>> nx, ny = (3, 2)
    >>> x = np.linspace(0, 1, nx)
    >>> y = np.linspace(0, 1, ny)
    >>> xv, yv = np.meshgrid(x, y)
    >>> xv
    array([[0. , 0.5, 1. ],
           [0. , 0.5, 1. ]])
    >>> yv
    array([[0.,  0.,  0.],
           [1.,  1.,  1.]])

    The result of `meshgrid` is a coordinate grid:

    >>> import matplotlib.pyplot as plt
    >>> plt.plot(xv, yv, marker='o', color='k', linestyle='none')
    >>> plt.show()

    You can create sparse output arrays to save memory and computation time.

    >>> xv, yv = np.meshgrid(x, y, sparse=True)
    >>> xv
    array([[0. ,  0.5,  1. ]])
    >>> yv
    array([[0.],
           [1.]])

    `meshgrid` is very useful to evaluate functions on a grid. If the
    function depends on all coordinates, both dense and sparse outputs can be
    used.

    >>> x = np.linspace(-5, 5, 101)
    >>> y = np.linspace(-5, 5, 101)
    >>> # full coordinate arrays
    >>> xx, yy = np.meshgrid(x, y)
    >>> zz = np.sqrt(xx**2 + yy**2)
    >>> xx.shape, yy.shape, zz.shape
    ((101, 101), (101, 101), (101, 101))
    >>> # sparse coordinate arrays
    >>> xs, ys = np.meshgrid(x, y, sparse=True)
    >>> zs = np.sqrt(xs**2 + ys**2)
    >>> xs.shape, ys.shape, zs.shape
    ((1, 101), (101, 1), (101, 101))
    >>> np.array_equal(zz, zs)
    True

    >>> h = plt.contourf(x, y, zs)
    >>> plt.axis('scaled')
    >>> plt.colorbar()
    >>> plt.show()
    """
    return np.meshgrid(*args, **kwargs)


def integral(*args, **kwargs):
    """
    Compute a definite integral.

    Integrate func from `a` to `b` (possibly infinite interval) using a
    technique from the Fortran library QUADPACK.

    Parameters
    ----------
    func : {function, scipy.LowLevelCallable}
        A Python function or method to integrate. If `func` takes many
        arguments, it is integrated along the axis corresponding to the
        first argument.

        If the user desires improved integration performance, then `f` may
        be a `scipy.LowLevelCallable` with one of the signatures::

            double func(double x)
            double func(double x, void *user_data)
            double func(int n, double *xx)
            double func(int n, double *xx, void *user_data)

        The ``user_data`` is the data contained in the `scipy.LowLevelCallable`.
        In the call forms with ``xx``,  ``n`` is the length of the ``xx``
        array which contains ``xx[0] == x`` and the rest of the items are
        numbers contained in the ``args`` argument of quad.

        In addition, certain ctypes call signatures are supported for
        backward compatibility, but those should not be used in new code.
    a : float
        Lower limit of integration (use -numpy.inf for -infinity).
    b : float
        Upper limit of integration (use numpy.inf for +infinity).
    args : tuple, optional
        Extra arguments to pass to `func`.
    full_output : int, optional
        Non-zero to return a dictionary of integration information.
        If non-zero, warning messages are also suppressed and the
        message is appended to the output tuple.
    complex_func : bool, optional
        Indicate if the function's (`func`) return type is real
        (``complex_func=False``: default) or complex (``complex_func=True``).
        In both cases, the function's argument is real.
        If full_output is also non-zero, the `infodict`, `message`, and
        `explain` for the real and complex components are returned in
        a dictionary with keys "real output" and "imag output".

    Returns
    -------
    y : float
        The integral of func from `a` to `b`.
    abserr : float
        An estimate of the absolute error in the result.
    infodict : dict
        A dictionary containing additional information.
    message
        A convergence message.
    explain
        Appended only with 'cos' or 'sin' weighting and infinite
        integration limits, it contains an explanation of the codes in
        infodict['ierlst']

    Other Parameters
    ----------------
    epsabs : float or int, optional
        Absolute error tolerance. Default is 1.49e-8. `quad` tries to obtain
        an accuracy of ``abs(i-result) <= max(epsabs, epsrel*abs(i))``
        where ``i`` = integral of `func` from `a` to `b`, and ``result`` is the
        numerical approximation. See `epsrel` below.
    epsrel : float or int, optional
        Relative error tolerance. Default is 1.49e-8.
        If ``epsabs <= 0``, `epsrel` must be greater than both 5e-29
        and ``50 * (machine epsilon)``. See `epsabs` above.
    limit : float or int, optional
        An upper bound on the number of subintervals used in the adaptive
        algorithm.
    points : (sequence of floats,ints), optional
        A sequence of break points in the bounded integration interval
        where local difficulties of the integrand may occur (e.g.,
        singularities, discontinuities). The sequence does not have
        to be sorted. Note that this option cannot be used in conjunction
        with ``weight``.
    weight : float or int, optional
        String indicating weighting function. Full explanation for this
        and the remaining arguments can be found below.
    wvar : optional
        Variables for use with weighting functions.
    wopts : optional
        Optional input for reusing Chebyshev moments.
    maxp1 : float or int, optional
        An upper bound on the number of Chebyshev moments.
    limlst : int, optional
        Upper bound on the number of cycles (>=3) for use with a sinusoidal
        weighting and an infinite end-point.

    See Also
    --------
    dblquad : double integral
    tplquad : triple integral
    nquad : n-dimensional integrals (uses `quad` recursively)
    fixed_quad : fixed-order Gaussian quadrature
    quadrature : adaptive Gaussian quadrature
    odeint : ODE integrator
    ode : ODE integrator
    simpson : integrator for sampled data
    romb : integrator for sampled data
    scipy.special : for coefficients and roots of orthogonal polynomials

    Notes
    -----
    For valid results, the integral must converge; behavior for divergent
    integrals is not guaranteed.

    **Extra information for quad() inputs and outputs**

    If full_output is non-zero, then the third output argument
    (infodict) is a dictionary with entries as tabulated below. For
    infinite limits, the range is transformed to (0,1) and the
    optional outputs are given with respect to this transformed range.
    Let M be the input argument limit and let K be infodict['last'].
    The entries are:

    'neval'
        The number of function evaluations.
    'last'
        The number, K, of subintervals produced in the subdivision process.
    'alist'
        A rank-1 array of length M, the first K elements of which are the
        left end points of the subintervals in the partition of the
        integration range.
    'blist'
        A rank-1 array of length M, the first K elements of which are the
        right end points of the subintervals.
    'rlist'
        A rank-1 array of length M, the first K elements of which are the
        integral approximations on the subintervals.
    'elist'
        A rank-1 array of length M, the first K elements of which are the
        moduli of the absolute error estimates on the subintervals.
    'iord'
        A rank-1 integer array of length M, the first L elements of
        which are pointers to the error estimates over the subintervals
        with ``L=K`` if ``K<=M/2+2`` or ``L=M+1-K`` otherwise. Let I be the
        sequence ``infodict['iord']`` and let E be the sequence
        ``infodict['elist']``.  Then ``E[I[1]], ..., E[I[L]]`` forms a
        decreasing sequence.

    If the input argument points is provided (i.e., it is not None),
    the following additional outputs are placed in the output
    dictionary. Assume the points sequence is of length P.

    'pts'
        A rank-1 array of length P+2 containing the integration limits
        and the break points of the intervals in ascending order.
        This is an array giving the subintervals over which integration
        will occur.
    'level'
        A rank-1 integer array of length M (=limit), containing the
        subdivision levels of the subintervals, i.e., if (aa,bb) is a
        subinterval of ``(pts[1], pts[2])`` where ``pts[0]`` and ``pts[2]``
        are adjacent elements of ``infodict['pts']``, then (aa,bb) has level l
        if ``|bb-aa| = |pts[2]-pts[1]| * 2**(-l)``.
    'ndin'
        A rank-1 integer array of length P+2. After the first integration
        over the intervals (pts[1], pts[2]), the error estimates over some
        of the intervals may have been increased artificially in order to
        put their subdivision forward. This array has ones in slots
        corresponding to the subintervals for which this happens.

    **Weighting the integrand**

    The input variables, *weight* and *wvar*, are used to weight the
    integrand by a select list of functions. Different integration
    methods are used to compute the integral with these weighting
    functions, and these do not support specifying break points. The
    possible values of weight and the corresponding weighting functions are.

    ==========  ===================================   =====================
    ``weight``  Weight function used                  ``wvar``
    ==========  ===================================   =====================
    'cos'       cos(w*x)                              wvar = w
    'sin'       sin(w*x)                              wvar = w
    'alg'       g(x) = ((x-a)**alpha)*((b-x)**beta)   wvar = (alpha, beta)
    'alg-loga'  g(x)*log(x-a)                         wvar = (alpha, beta)
    'alg-logb'  g(x)*log(b-x)                         wvar = (alpha, beta)
    'alg-log'   g(x)*log(x-a)*log(b-x)                wvar = (alpha, beta)
    'cauchy'    1/(x-c)                               wvar = c
    ==========  ===================================   =====================

    wvar holds the parameter w, (alpha, beta), or c depending on the weight
    selected. In these expressions, a and b are the integration limits.

    For the 'cos' and 'sin' weighting, additional inputs and outputs are
    available.

    For finite integration limits, the integration is performed using a
    Clenshaw-Curtis method which uses Chebyshev moments. For repeated
    calculations, these moments are saved in the output dictionary:

    'momcom'
        The maximum level of Chebyshev moments that have been computed,
        i.e., if ``M_c`` is ``infodict['momcom']`` then the moments have been
        computed for intervals of length ``|b-a| * 2**(-l)``,
        ``l=0,1,...,M_c``.
    'nnlog'
        A rank-1 integer array of length M(=limit), containing the
        subdivision levels of the subintervals, i.e., an element of this
        array is equal to l if the corresponding subinterval is
        ``|b-a|* 2**(-l)``.
    'chebmo'
        A rank-2 array of shape (25, maxp1) containing the computed
        Chebyshev moments. These can be passed on to an integration
        over the same interval by passing this array as the second
        element of the sequence wopts and passing infodict['momcom'] as
        the first element.

    If one of the integration limits is infinite, then a Fourier integral is
    computed (assuming w neq 0). If full_output is 1 and a numerical error
    is encountered, besides the error message attached to the output tuple,
    a dictionary is also appended to the output tuple which translates the
    error codes in the array ``info['ierlst']`` to English messages. The
    output information dictionary contains the following entries instead of
    'last', 'alist', 'blist', 'rlist', and 'elist':

    'lst'
        The number of subintervals needed for the integration (call it ``K_f``).
    'rslst'
        A rank-1 array of length M_f=limlst, whose first ``K_f`` elements
        contain the integral contribution over the interval
        ``(a+(k-1)c, a+kc)`` where ``c = (2*floor(|w|) + 1) * pi / |w|``
        and ``k=1,2,...,K_f``.
    'erlst'
        A rank-1 array of length ``M_f`` containing the error estimate
        corresponding to the interval in the same position in
        ``infodict['rslist']``.
    'ierlst'
        A rank-1 integer array of length ``M_f`` containing an error flag
        corresponding to the interval in the same position in
        ``infodict['rslist']``.  See the explanation dictionary (last entry
        in the output tuple) for the meaning of the codes.


    **Details of QUADPACK level routines**

    `quad` calls routines from the FORTRAN library QUADPACK. This section
    provides details on the conditions for each routine to be called and a
    short description of each routine. The routine called depends on
    `weight`, `points` and the integration limits `a` and `b`.

    ================  ==============  ==========  =====================
    QUADPACK routine  `weight`        `points`    infinite bounds
    ================  ==============  ==========  =====================
    qagse             None            No          No
    qagie             None            No          Yes
    qagpe             None            Yes         No
    qawoe             'sin', 'cos'    No          No
    qawfe             'sin', 'cos'    No          either `a` or `b`
    qawse             'alg*'          No          No
    qawce             'cauchy'        No          No
    ================  ==============  ==========  =====================

    The following provides a short description from [1]_ for each
    routine.

    qagse
        is an integrator based on globally adaptive interval
        subdivision in connection with extrapolation, which will
        eliminate the effects of integrand singularities of
        several types.
    qagie
        handles integration over infinite intervals. The infinite range is
        mapped onto a finite interval and subsequently the same strategy as
        in ``QAGS`` is applied.
    qagpe
        serves the same purposes as QAGS, but also allows the
        user to provide explicit information about the location
        and type of trouble-spots i.e. the abscissae of internal
        singularities, discontinuities and other difficulties of
        the integrand function.
    qawoe
        is an integrator for the evaluation of
        :math:`\\int^b_a \\cos(\\omega x)f(x)dx` or
        :math:`\\int^b_a \\sin(\\omega x)f(x)dx`
        over a finite interval [a,b], where :math:`\\omega` and :math:`f`
        are specified by the user. The rule evaluation component is based
        on the modified Clenshaw-Curtis technique

        An adaptive subdivision scheme is used in connection
        with an extrapolation procedure, which is a modification
        of that in ``QAGS`` and allows the algorithm to deal with
        singularities in :math:`f(x)`.
    qawfe
        calculates the Fourier transform
        :math:`\\int^\\infty_a \\cos(\\omega x)f(x)dx` or
        :math:`\\int^\\infty_a \\sin(\\omega x)f(x)dx`
        for user-provided :math:`\\omega` and :math:`f`. The procedure of
        ``QAWO`` is applied on successive finite intervals, and convergence
        acceleration by means of the :math:`\\varepsilon`-algorithm is applied
        to the series of integral approximations.
    qawse
        approximate :math:`\\int^b_a w(x)f(x)dx`, with :math:`a < b` where
        :math:`w(x) = (x-a)^{\\alpha}(b-x)^{\\beta}v(x)` with
        :math:`\\alpha,\\beta > -1`, where :math:`v(x)` may be one of the
        following functions: :math:`1`, :math:`\\log(x-a)`, :math:`\\log(b-x)`,
        :math:`\\log(x-a)\\log(b-x)`.

        The user specifies :math:`\\alpha`, :math:`\\beta` and the type of the
        function :math:`v`. A globally adaptive subdivision strategy is
        applied, with modified Clenshaw-Curtis integration on those
        subintervals which contain `a` or `b`.
    qawce
        compute :math:`\\int^b_a f(x) / (x-c)dx` where the integral must be
        interpreted as a Cauchy principal value integral, for user specified
        :math:`c` and :math:`f`. The strategy is globally adaptive. Modified
        Clenshaw-Curtis integration is used on those intervals containing the
        point :math:`x = c`.

    **Integration of Complex Function of a Real Variable**

    A complex valued function, :math:`f`, of a real variable can be written as
    :math:`f = g + ih`.  Similarly, the integral of :math:`f` can be
    written as

    .. math::
        \\int_a^b f(x) dx = \\int_a^b g(x) dx + i\\int_a^b h(x) dx

    assuming that the integrals of :math:`g` and :math:`h` exist
    over the interval :math:`[a,b]` [2]_. Therefore, ``quad`` integrates
    complex-valued functions by integrating the real and imaginary components
    separately.


    References
    ----------

    .. [1] Piessens, Robert; de Doncker-Kapenga, Elise;
           Überhuber, Christoph W.; Kahaner, David (1983).
           QUADPACK: A subroutine package for automatic integration.
           Springer-Verlag.
           ISBN 978-3-540-12553-2.

    .. [2] McCullough, Thomas; Phillips, Keith (1973).
           Foundations of Analysis in the Complex Plane.
           Holt Rinehart Winston.
           ISBN 0-03-086370-8

    Examples
    --------
    Calculate :math:`\\int^4_0 x^2 dx` and compare with an analytic result

    >>> from scipy import integrate
    >>> import numpy as np
    >>> x2 = lambda x: x**2
    >>> integrate.quad(x2, 0, 4)
    (21.333333333333332, 2.3684757858670003e-13)
    >>> print(4**3 / 3.)  # analytical result
    21.3333333333

    Calculate :math:`\\int^\\infty_0 e^{-x} dx`

    >>> invexp = lambda x: np.exp(-x)
    >>> integrate.quad(invexp, 0, np.inf)
    (1.0, 5.842605999138044e-11)

    Calculate :math:`\\int^1_0 a x \\,dx` for :math:`a = 1, 3`

    >>> f = lambda x, a: a*x
    >>> y, err = integrate.quad(f, 0, 1, args=(1,))
    >>> y
    0.5
    >>> y, err = integrate.quad(f, 0, 1, args=(3,))
    >>> y
    1.5

    Calculate :math:`\\int^1_0 x^2 + y^2 dx` with ctypes, holding
    y parameter as 1::

        testlib.c =>
            double func(int n, double args[n]){
                return args[0]*args[0] + args[1]*args[1];}
        compile to library testlib.*

    ::

       from scipy import integrate
       import ctypes
       lib = ctypes.CDLL('/home/.../testlib.*') #use absolute path
       lib.func.restype = ctypes.c_double
       lib.func.argtypes = (ctypes.c_int,ctypes.c_double)
       integrate.quad(lib.func,0,1,(1))
       #(1.3333333333333333, 1.4802973661668752e-14)
       print((1.0**3/3.0 + 1.0) - (0.0**3/3.0 + 0.0)) #Analytic result
       # 1.3333333333333333

    Be aware that pulse shapes and other sharp features as compared to the
    size of the integration interval may not be integrated correctly using
    this method. A simplified example of this limitation is integrating a
    y-axis reflected step function with many zero values within the integrals
    bounds.

    >>> y = lambda x: 1 if x<=0 else 0
    >>> integrate.quad(y, -1, 1)
    (1.0, 1.1102230246251565e-14)
    >>> integrate.quad(y, -1, 100)
    (1.0000000002199108, 1.0189464580163188e-08)
    >>> integrate.quad(y, -1, 10000)
    (0.0, 0.0)

    """
    i, _ = quad(*args, **kwargs)
    return i


def integral2(*args, **kwargs):
    """
    Compute a double integral.

    Return the double (definite) integral of ``func(y, x)`` from ``x = a..b``
    and ``y = gfun(x)..hfun(x)``.

    Parameters
    ----------
    func : callable
        A Python function or method of at least two variables: y must be the
        first argument and x the second argument.
    a, b : float
        The limits of integration in x: `a` < `b`
    gfun : callable or float
        The lower boundary curve in y which is a function taking a single
        floating point argument (x) and returning a floating point result
        or a float indicating a constant boundary curve.
    hfun : callable or float
        The upper boundary curve in y (same requirements as `gfun`).
    args : sequence, optional
        Extra arguments to pass to `func`.
    epsabs : float, optional
        Absolute tolerance passed directly to the inner 1-D quadrature
        integration. Default is 1.49e-8. ``dblquad`` tries to obtain
        an accuracy of ``abs(i-result) <= max(epsabs, epsrel*abs(i))``
        where ``i`` = inner integral of ``func(y, x)`` from ``gfun(x)``
        to ``hfun(x)``, and ``result`` is the numerical approximation.
        See `epsrel` below.
    epsrel : float, optional
        Relative tolerance of the inner 1-D integrals. Default is 1.49e-8.
        If ``epsabs <= 0``, `epsrel` must be greater than both 5e-29
        and ``50 * (machine epsilon)``. See `epsabs` above.

    Returns
    -------
    y : float
        The resultant integral.
    abserr : float
        An estimate of the error.

    See Also
    --------
    quad : single integral
    tplquad : triple integral
    nquad : N-dimensional integrals
    fixed_quad : fixed-order Gaussian quadrature
    quadrature : adaptive Gaussian quadrature
    odeint : ODE integrator
    ode : ODE integrator
    simpson : integrator for sampled data
    romb : integrator for sampled data
    scipy.special : for coefficients and roots of orthogonal polynomials


    Notes
    -----
    For valid results, the integral must converge; behavior for divergent
    integrals is not guaranteed.

    **Details of QUADPACK level routines**

    `quad` calls routines from the FORTRAN library QUADPACK. This section
    provides details on the conditions for each routine to be called and a
    short description of each routine. For each level of integration, ``qagse``
    is used for finite limits or ``qagie`` is used if either limit (or both!)
    are infinite. The following provides a short description from [1]_ for each
    routine.

    qagse
        is an integrator based on globally adaptive interval
        subdivision in connection with extrapolation, which will
        eliminate the effects of integrand singularities of
        several types.
    qagie
        handles integration over infinite intervals. The infinite range is
        mapped onto a finite interval and subsequently the same strategy as
        in ``QAGS`` is applied.

    References
    ----------

    .. [1] Piessens, Robert; de Doncker-Kapenga, Elise;
           Überhuber, Christoph W.; Kahaner, David (1983).
           QUADPACK: A subroutine package for automatic integration.
           Springer-Verlag.
           ISBN 978-3-540-12553-2.

    Examples
    --------
    Compute the double integral of ``x * y**2`` over the box
    ``x`` ranging from 0 to 2 and ``y`` ranging from 0 to 1.
    That is, :math:`\\int^{x=2}_{x=0} \\int^{y=1}_{y=0} x y^2 \\,dy \\,dx`.

    >>> import numpy as np
    >>> from scipy import integrate
    >>> f = lambda y, x: x*y**2
    >>> integrate.dblquad(f, 0, 2, 0, 1)
        (0.6666666666666667, 7.401486830834377e-15)

    Calculate :math:`\\int^{x=\\pi/4}_{x=0} \\int^{y=\\cos(x)}_{y=\\sin(x)} 1
    \\,dy \\,dx`.

    >>> f = lambda y, x: 1
    >>> integrate.dblquad(f, 0, np.pi/4, np.sin, np.cos)
        (0.41421356237309503, 1.1083280054755938e-14)

    Calculate :math:`\\int^{x=1}_{x=0} \\int^{y=2-x}_{y=x} a x y \\,dy \\,dx`
    for :math:`a=1, 3`.

    >>> f = lambda y, x, a: a*x*y
    >>> integrate.dblquad(f, 0, 1, lambda x: x, lambda x: 2-x, args=(1,))
        (0.33333333333333337, 5.551115123125783e-15)
    >>> integrate.dblquad(f, 0, 1, lambda x: x, lambda x: 2-x, args=(3,))
        (0.9999999999999999, 1.6653345369377348e-14)

    Compute the two-dimensional Gaussian Integral, which is the integral of the
    Gaussian function :math:`f(x,y) = e^{-(x^{2} + y^{2})}`, over
    :math:`(-\\infty,+\\infty)`. That is, compute the integral
    :math:`\\iint^{+\\infty}_{-\\infty} e^{-(x^{2} + y^{2})} \\,dy\\,dx`.

    >>> f = lambda x, y: np.exp(-(x ** 2 + y ** 2))
    >>> integrate.dblquad(f, -np.inf, np.inf, -np.inf, np.inf)
        (3.141592653589777, 2.5173086737433208e-08)

    """
    i, _ = dblquad(*args, **kwargs)
    return i


def randperm(n):
    """
    Return a random permutation of an array.

    Parameters
    ----------
    n : int
        Length of the array.

    Returns
    -------
    out : ndarray
        Random permutation of the array.
    """
    return np.random.permutation(n) + 1


def randi(range, *size):
    """
        Generates random integers based on the given range and size.

        Parameters
            - range (int or list): Defines the range for generating random integers.
                - If an integer is provided, random integers are generated between 1 and the given integer (inclusive).
                - If a list is provided:
                    - If it has two elements, random integers are generated between the first and second element (inclusive).
                    - If it has one element, random integers are generated between 1 and the element (inclusive).
                    - If the list has more than two elements, an error message is displayed.
            - size (int or tuple of ints): Specifies the number of random integers to generate or the shape of the array to return.

        Returns
            - np.ndarray: An array of random integers based on the specified range and size.
            - None: If the input is invalid.

        Raises
            - Prints error messages if `range` is not an integer or a list with one or two elements.
    """
    # checks if range is a list
    if isinstance(range, list):
        # checks if it has two elements
        if len(range) == 2:
            # returns the random integer
            return np.random.randint(range[0], range[1] + 1, size)
        elif len(range) == 1:
            # returns the random integer
            return np.random.randint(1, range[0] + 1, size)
        else:
            print("Error: range must have one or two elements")
            return None
    elif isinstance(range, int):
        return np.random.randint(1, range + 1, size)
    else:
        print("Error: range must be an integer or a list")

def rand(*args, **kwargs):
    """
    Random values in a given shape.
    
    Create an array of the given shape and populate it with random
    samples from a uniform distribution over [0, 1).

    Parameters

        d0, d1, ..., dn : int, optional
            The dimensions of the returned array, must be non-negative. 
            If no argument is given a single Python float is returned.

    Returns

        out : ndarray, shape (d0, d1, ..., dn) Random values.
    """
    return np.random.rand(*args, **kwargs)


def normrnd(mu, sigma, *size):
    """
        Draw random samples from a normal (Gaussian) distribution.

        The probability density function of the normal distribution, 
        first derived by De Moivre and 200 years later by both Gauss and 
        Laplace independently 2, is often called the bell curve because of 
        its characteristic shape (see the example below).

        The normal distributions occurs often in nature. 
        For example, it describes the commonly occurring distribution of 
        samples influenced by a large number of tiny, random disturbances, 
        each with its own unique distribution 2.

        Parameters

            loc : float or array_like of floats Mean ("centre") of the distribution.

            scale : float or array_like of floats
                Standard deviation (spread or "width") of the distribution. Must be non-negative.

            size : int or tuple of ints, optional
                Output shape. If the given shape is, e.g., (m, n, k), 
                then m * n * k samples are drawn. If size is None (default), 
                a single value is returned if loc and scale are both scalars. 
                Otherwise, np.broadcast(loc, scale).size samples are drawn.

        Returns

            out : ndarray or scalar
            Drawn samples from the parameterized normal distribution.
        
    """
    return np.random.normal(mu, sigma, size)


def chi2rnd(nu, *size):
    """
        Draw samples from a chi-square distribution.

        When `df` independent random variables, each with standard normal 
        distributions (mean 0, variance 1), are squared and summed, the 
        resulting distribution is chi-square (see Notes). This distribution 
        is often used in hypothesis testing.

        Parameters

            df : int or array_like of ints
                Number of degrees of freedom, must be > 0.

            size : int or tuple of ints, optional
                Output shape. If the given shape is, e.g., (m, n, k), 
                then m * n * k samples are drawn. If size is None (default), 
                a single value is returned if `df` is a scalar. Otherwise, 
                np.array(df).size samples are drawn.

        Returns

            out : ndarray or scalar
                Drawn samples from the parameterized chi-square distribution.
    """
    return np.random.chisquare(nu, size)
