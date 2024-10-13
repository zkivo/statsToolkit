import os
import matplotlib.pyplot as plt
from statstoolkit.visualization import bar_chart, pie_chart, histogram, boxplot, scatterplot

# Create the examples directory if it doesn't exist
os.makedirs('examples', exist_ok=True)

# 1. Save Bar Chart
categories = ['Category A', 'Category B', 'Category C']
values = [10, 20, 30]
plt.figure()
bar_chart(categories, values, title="Example Bar Chart", xlabel="Category", ylabel="Values", color="blue")
plt.savefig('examples/bar_chart.png')
plt.close()

# 2. Save Pie Chart
sizes = [15, 30, 45, 10]
labels = ['Category A', 'Category B', 'Category C', 'Category D']
plt.figure()
pie_chart(sizes, labels=labels, title="Example Pie Chart", autopct='%1.1f%%', shadow=True)
plt.savefig('examples/pie_chart.png')
plt.close()

# 3. Save Histogram
data = [1, 2, 2, 3, 3, 3, 4, 4, 4, 4]
plt.figure()
histogram(data, bins=4, title="Example Histogram", xlabel="Value", ylabel="Frequency", color="green")
plt.savefig('examples/histogram.png')
plt.close()

# 4. Save Boxplot
MPG = [15, 18, 21, 24, 30]
origin = ['USA', 'Japan', 'Japan', 'USA', 'Europe']
plt.figure()
boxplot(MPG, origin=origin, title="Example Boxplot by Car Origin", xlabel="Origin", ylabel="Miles per Gallon")
plt.savefig('examples/boxplot.png')
plt.close()

# 5. Save 2D Scatter Plot
x = [1, 2, 3, 4]
y = [10, 20, 30, 40]
plt.figure()
scatterplot(x, y, title="Example 2D Scatter Plot", xlabel="X-Axis", ylabel="Y-Axis", color="red")
plt.savefig('examples/scatterplot_2d.png')
plt.close()

# 6. Save 3D-like Scatter Plot
z = [50, 100, 200, 300]
plt.figure()
scatterplot(x, y, z=z, symbol="*", title="Example 3D-like Scatter Plot", xlabel="X-Axis", ylabel="Y-Axis", color="blue")
plt.savefig('examples/scatterplot_3d.png')
plt.close()
