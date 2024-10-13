import matplotlib.pyplot as plt
import seaborn as sns


def bar_chart(x, y, title=None, xlabel=None, ylabel=None, color=None, figsize=(10, 6), **kwargs):
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


def pie_chart(sizes, labels=None, title=None, colors=None, explode=None, autopct='%1.1f%%', shadow=False, startangle=90,
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


def histogram(x, bins=10, title=None, xlabel=None, ylabel=None, color=None, figsize=(10, 6), **kwargs):
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


def scatterplot(x, y, z=None, symbol='o', title=None, xlabel=None, ylabel=None, color=None, figsize=(10, 6), **kwargs):
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
