import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


def bar(x, y, title=None, xlabel=None, ylabel=None, color=None, figsize=(10, 6), **kwargs):
    """
    Create a bar chart.

    Parameters
    ----------
    x : list or array-like
        The categories or labels for the x-axis.
    y : list or array-like
        The values for each category.
    title : str, optional
        Title of the chart.
    xlabel : str, optional
        Label for the x-axis.
    ylabel : str, optional
        Label for the y-axis.
    color : str or list, optional
        Color(s) for the bars. Can be a single color or a list of colors for each bar.
    figsize : tuple, optional
        Size of the figure (width, height) in inches.
    kwargs : additional keyword arguments
        Additional arguments for `matplotlib.pyplot.bar`.

    Examples
    --------
    >>> bar_chart(['A', 'B', 'C'], [10, 20, 30], title="Bar Chart", xlabel="Category", ylabel="Value")
    """
    plt.figure(figsize=figsize)
    plt.bar(x, y, color=color, **kwargs)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()


def pie(sizes, labels=None, title=None, colors=None, explode=None, autopct='%1.1f%%', shadow=False, startangle=90,
              **kwargs):
    """
    Create a pie chart.

    Parameters
    ----------
    sizes : list or array-like
        The size of each wedge.
    labels : list, optional
        A sequence of strings providing the labels for each wedge.
    title : str, optional
        Title of the chart.
    colors : list, optional
        A sequence of colors for the wedges.
    explode : list, optional
        A sequence of floats that indicates the fraction of the radius with which to offset each wedge.
    autopct : str, optional
        String or function used to label the wedges with their numeric value.
    shadow : bool, optional
        Whether to draw a shadow beneath the pie chart.
    startangle : int, optional
        Starting angle for the pie chart, default is 90 degrees.
    kwargs : additional keyword arguments
        Additional arguments for `matplotlib.pyplot.pie`.

    Examples
    --------
    >>> pie_chart([15, 30, 45, 10], labels=['A', 'B', 'C', 'D'], title="Pie Chart")
    """
    plt.figure(figsize=(8, 8))
    plt.pie(sizes, labels=labels, colors=colors, explode=explode, autopct=autopct, shadow=shadow, startangle=startangle,
            **kwargs)
    plt.title(title)
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.show()


def hist(x, bins=10, title=None, xlabel=None, ylabel=None, color=None, figsize=(10, 6), **kwargs):
    """
    Create a histogram.

    Parameters
    ----------
    x : list or array-like
        Data to plot in the histogram.
    bins : int or list, optional
        Number of bins or bin edges for the histogram.
    title : str, optional
        Title of the chart.
    xlabel : str, optional
        Label for the x-axis.
    ylabel : str, optional
        Label for the y-axis.
    color : str, optional
        Color of the bars.
    figsize : tuple, optional
        Size of the figure (width, height) in inches.
    kwargs : additional keyword arguments
        Additional arguments for `matplotlib.pyplot.hist`.

    Examples
    --------
    >>> histogram([1, 2, 2, 3, 3, 3, 4, 4, 4, 4], bins=4, title="Histogram", xlabel="Value", ylabel="Frequency")
    """
    plt.figure(figsize=figsize)
    plt.hist(x, bins=bins, color=color, **kwargs)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()


def boxplot(MPG, origin=None, title=None, xlabel=None, ylabel=None, color=None, figsize=(10, 6), **kwargs):
    """
    Create a boxplot.

    Parameters
    ----------
    MPG : list or array-like
        Data to plot in the boxplot (e.g., miles per gallon).
    origin : list or array-like, optional
        Grouping variable for the boxplot (e.g., origin of cars).
    title : str, optional
        Title of the chart.
    xlabel : str, optional
        Label for the x-axis.
    ylabel : str, optional
        Label for the y-axis.
    color : str, optional
        Color of the boxplot.
    figsize : tuple, optional
        Size of the figure (width, height) in inches.
    kwargs : additional keyword arguments
        Additional arguments for `seaborn.boxplot`.

    Examples
    --------
    >>> boxplot([15, 18, 21, 24, 30], title="MPG Boxplot", ylabel="Miles per Gallon")
    >>> boxplot([15, 18, 21, 24, 30], origin=['USA', 'USA', 'Japan', 'Europe', 'Japan'], title="MPG by Origin")
    """
    plt.figure(figsize=figsize)
    if origin is None:
        sns.boxplot(x=MPG, color=color, **kwargs)
    else:
        sns.boxplot(x=origin, y=MPG, color=color, **kwargs)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()


def scatter_2d(x, y, z=None, symbol='o', title=None, xlabel=None, ylabel=None, color=None, figsize=(10, 6), **kwargs):
    """
    Create a scatter plot.

    Parameters
    ----------
    x : list or array-like
        Data for the x-axis.
    y : list or array-like
        Data for the y-axis.
    z : list or array-like, optional
        Data for color mapping or size mapping in 3D scatter plot.
    symbol : str, optional
        Marker symbol for the scatter plot (default is 'o').
    title : str, optional
        Title of the chart.
    xlabel : str, optional
        Label for the x-axis.
    ylabel : str, optional
        Label for the y-axis.
    color : str or list, optional
        Color(s) for the points. Can be a single color or a list of colors for each point.
    figsize : tuple, optional
        Size of the figure (width, height) in inches.
    kwargs : additional keyword arguments
        Additional arguments for `matplotlib.pyplot.scatter`.

    Examples
    --------
    >>> scatterplot([1, 2, 3, 4], [10, 20, 25, 30], title="Scatter Plot", xlabel="X", ylabel="Y")
    >>> scatterplot([1, 2, 3, 4], [10, 20, 25, 30], z=[50, 100, 200, 300], symbol="*", title="3D Scatter Plot")
    """
    plt.figure(figsize=figsize)

    if z is None:
        plt.scatter(x, y, marker=symbol, c=color, **kwargs)
    else:
        plt.scatter(x, y, s=z, marker=symbol, c=color, **kwargs)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()


def scatter(x, y, z = None, symbol='o', title=None, xlabel=None, ylabel=None, zlabel=None, color=None, figsize=(10, 6),
                   **kwargs):
    """
    Create a true 3D scatter plot.

    Parameters
    ----------
    x : list or array-like
        Data for the x-axis.
    y : list or array-like
        Data for the y-axis.
    z : list or array-like
        Data for the z-axis (third dimension).
    symbol : str, optional
        Marker symbol for the scatter plot (default is 'o').
    title : str, optional
        Title of the chart.
    xlabel : str, optional
        Label for the x-axis.
    ylabel : str, optional
        Label for the y-axis.
    zlabel : str, optional
        Label for the z-axis.
    color : str or list, optional
        Color(s) for the points.
    figsize : tuple, optional
        Size of the figure (width, height) in inches.
    kwargs : additional keyword arguments
        Additional arguments for `Axes3D.scatter`.

    Examples
    --------
    >>> scatterplot_3d([1, 2, 3, 4], [10, 20, 25, 30], [50, 100, 200, 300], title="3D Scatter Plot")
    """

    if z is None:
        scatter_2d(x, y, symbol=symbol, title=title, xlabel=xlabel, ylabel=ylabel, color=color, figsize=figsize, **kwargs)
        return

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')

    # Plot the 3D scatter
    ax.scatter(x, y, z, marker=symbol, c=color, **kwargs)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    plt.show()


def plot_regression_surface(x1, x2, y, b):
    """
    Generate a 3D scatter plot with regression plane.

    Parameters:
    x1 : array-like
        Predictor variable 1 (e.g., weight).
    x2 : array-like
        Predictor variable 2 (e.g., horsepower).
    y : array-like
        Response variable (e.g., MPG).
    b : array
        Coefficients from the regression.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x1, x2, y, color='b', s=50)

    # Generate the grid for the regression plane
    x1fit = np.linspace(min(x1), max(x1), 20)
    x2fit = np.linspace(min(x2), max(x2), 20)
    X1FIT, X2FIT = np.meshgrid(x1fit, x2fit)
    YFIT = b[0] + b[1] * X1FIT + b[2] * X2FIT + b[3] * X1FIT * X2FIT

    # Plot the surface
    ax.plot_surface(X1FIT, X2FIT, YFIT, color='c', alpha=0.5, edgecolor='none')

    # Labels and view
    ax.set_xlabel('Weight')
    ax.set_ylabel('Horsepower')
    ax.set_zlabel('MPG')
    ax.view_init(elev=10, azim=50)
    plt.show()