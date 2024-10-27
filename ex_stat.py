from statstoolkit.statistics import *
from statstoolkit.visualization import *
import numpy as np

# 1. Display Bar Chart
categories = ['Category A', 'Category B', 'Category C']
values = [10, 20, 30]
bar_chart(categories, values, title="Example Bar Chart", xlabel="Category", ylabel="Values", color="blue")

# 2. Display Pie Chart
sizes = [15, 30, 45, 10]
labels = ['Category A', 'Category B', 'Category C', 'Category D']
pie_chart(sizes, labels=labels, title="Example Pie Chart", autopct='%1.1f%%', shadow=True)

# 3. Display Histogram
data = [1, 2, 2, 3, 3, 3, 4, 4, 4, 4]
histogram(data, bins=4, title="Example Histogram", xlabel="Value", ylabel="Frequency", color="green")

# 4. Display Boxplot
MPG = [15, 18, 21, 24, 30]
origin = ['USA', 'Japan', 'Japan', 'USA', 'Europe']
boxplot(MPG, origin=origin, title="Example Boxplot by Car Origin", xlabel="Origin", ylabel="Miles per Gallon")

# 5. Display 2D Scatter Plot
x = [1, 2, 3, 4]
y = [10, 20, 30, 40]
scatterplot(x, y, title="Example 2D Scatter Plot", xlabel="X-Axis", ylabel="Y-Axis", color="red")

# 6. Display True 3D Scatter Plot
z = [50, 100, 200, 300]
scatterplot_3d(x, y, z, symbol="o", title="Example 3D Scatter Plot", xlabel="X-Axis", ylabel="Y-Axis", zlabel="Z-Axis", color="blue")

# 7. Display Linear Correlation (Pearson correlation)
a = np.array([1,       0,  -1,    3,    5, -2, 0.5])
b = np.array([-1,      2,   4, -0.5,    1,  1,   0])
c = np.array([-0.4,  1.2,   0,    3,  2.5, -1,   6])

print(corrcoef(a, b))

# Display Covariance

abc = np.stack((a, b, c), axis=1)

print(cov(a, b))
print(cov(abc))

# 8 Display Linear Regression
x = np.array([1,      0,    -1,   3,    5,   -2,  0.5])
y = np.array([1.65, 0.2, -1.69, 4.7, 7.57, -3.2, 0.65])

mdl = fitlm(x, y)

import pprint

def pretty_print(d):
    formatted_dict = {}
    for key, value in d.items():
        if isinstance(value, pd.DataFrame):
            formatted_dict[key] = value.to_string()  # Convert DataFrame to a string without index
        else:
            formatted_dict[key] = value
    pprint.pprint(formatted_dict)

pretty_print(mdl)

# 9. Multiple Linear Regression
x1 = np.array([1,      0,  -1,    3,   5, -2, 0.5])
x2 = np.array([-1,     2,   4, -0.5,   1,  1,   0])
x3 = np.array([-0.4, 1.2,   0,    3, 2.5, -1,   6])

y = np.array([4.49, 4.03, -2.04, 9.93, 10.78, 0.23, 15.2])
X = np.column_stack((np.ones_like(x1), x1, x2, x3))

b, bint, r, rint, _stats = regress(y, X)

print("b: ", b)
print("bint: ", bint)
print("r: ", r)
print("rint: ", rint)
pprint.pp(_stats)

# 10. One sample ttest
x = np.array([1, 0, -1, 3, 5, -2, 0.5])

h, p, ci, _stats = ttest(x)

print("h: ", h)
print("p: ", p)
print("ci: ", ci)
pprint.pp(_stats)

# added alpha and mu

h, p, ci, _stats = ttest(x, m = 0.8, alpha=0.01)

print("h: ", h)
print("p: ", p)
print("ci: ", ci)
pprint.pp(_stats)