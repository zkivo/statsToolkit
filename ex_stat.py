from statstoolkit.visualization import bar_chart, pie_chart, histogram, boxplot, scatterplot
from statstoolkit.visualization import scatterplot_3d  # Import the true 3D scatter plot

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
