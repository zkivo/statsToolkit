import os
import matplotlib.pyplot as plt
from statstoolkit.visualization import bar_chart, pie_chart, histogram, boxplot, scatterplot
from statstoolkit.visualization import scatterplot_3d  # Import the true 3D scatter plot

# Create the examples directory if it doesn't exist
os.makedirs('examples', exist_ok=True)

# 1. Display Bar Chart
categories = ['Category A', 'Category B', 'Category C']
values = [10, 20, 30]
plt.figure()
bar_chart(categories, values, title="Example Bar Chart", xlabel="Category", ylabel="Values", color="blue")
plt.show()  # Manually save this as 'bar_chart.png'

# 2. Display Pie Chart
sizes = [15, 30, 45, 10]
labels = ['Category A', 'Category B', 'Category C', 'Category D']
plt.figure()
pie_chart(sizes, labels=labels, title="Example Pie Chart", autopct='%1.1f%%', shadow=True)
plt.show()  # Manually save this as 'pie_chart.png'

# 3. Display Histogram
data = [1, 2, 2, 3, 3, 3, 4, 4, 4, 4]
plt.figure()
histogram(data, bins=4, title="Example Histogram", xlabel="Value", ylabel="Frequency", color="green")
plt.show()  # Manually save this as 'histogram.png'

# 4. Display Boxplot
MPG = [15, 18, 21, 24, 30]
origin = ['USA', 'Japan', 'Japan', 'USA', 'Europe']
plt.figure()
boxplot(MPG, origin=origin, title="Example Boxplot by Car Origin", xlabel="Origin", ylabel="Miles per Gallon")
plt.show()  # Manually save this as 'boxplot.png'

# 5. Display 2D Scatter Plot
x = [1, 2, 3, 4]
y = [10, 20, 30, 40]
plt.figure()
scatterplot(x, y, title="Example 2D Scatter Plot", xlabel="X-Axis", ylabel="Y-Axis", color="red")
plt.show()  # Manually save this as 'scatterplot_2d.png'

# 6. Display True 3D Scatter Plot
z = [50, 100, 200, 300]
plt.figure()
scatterplot_3d(x, y, z, symbol="o", title="Example 3D Scatter Plot", xlabel="X-Axis", ylabel="Y-Axis", zlabel="Z-Axis", color="blue")
plt.show()  # Manually save this as 'scatterplot_3d.png'
