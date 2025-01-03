from statstoolkit.statistics import *
from statstoolkit.visualization import *
from statstoolkit.utils import *

# 1. Display Bar Chart
categories = ['Category A', 'Category B', 'Category C']
values = [10, 20, 30]
bar(categories, values, title="Example Bar Chart", xlabel="Category", ylabel="Values", color="blue")

# 2. Display Pie Chart
sizes = [15, 30, 45, 10]
labels = ['Category A', 'Category B', 'Category C', 'Category D']
pie(sizes, labels=labels, title="Example Pie Chart", autopct='%1.1f%%', shadow=True)

# 3. Display Histogram
data = [1, 2, 2, 3, 3, 3, 4, 4, 4, 4]
hist(data, bins=4, title="Example Histogram", xlabel="Value", ylabel="Frequency", color="green")

# 4. Display Boxplot
MPG = [15, 18, 21, 24, 30]
origin = ['USA', 'Japan', 'Japan', 'USA', 'Europe']
boxplot(MPG, origin=origin, title="Example Boxplot by Car Origin", xlabel="Origin", ylabel="Miles per Gallon")

# 5. Display 2D Scatter Plot
x = [1, 2, 3, 4]
y = [10, 20, 30, 40]
scatter(x, y, title="Example 2D Scatter Plot", xlabel="X-Axis", ylabel="Y-Axis", color="red")

# 6. Display True 3D Scatter Plot
z = [50, 100, 200, 300]
scatter(x, y, z, symbol="o", title="Example 3D Scatter Plot", xlabel="X-Axis", ylabel="Y-Axis", zlabel="Z-Axis", color="blue")

print()
print("---------------------------------------------")
print("------------- Linear correlation -------------")
print("---------------------------------------------")
print()

a = np.array([1,       0,  -1,    3,    5, -2, 0.5])
b = np.array([-1,      2,   4, -0.5,    1,  1,   0])
c = np.array([-0.4,  1.2,   0,    3,  2.5, -1,   6])

R, P = corrcoef(a, b)
print('R:')
print(R)
print('P:')
print(P)

print()
print("---------------------------------------------")
print("------------- Covariance -------------")
print("---------------------------------------------")
print()

abc = np.stack((a, b, c), axis=1)

print("a, b: ")
print(cov(a, b))
print("a, b, c:")
print(cov(abc))

print()
print("---------------------------------------------")
print("------------- Linear regression -------------")
print("---------------------------------------------")
print()

x = np.array([1,      0,    -1,   3,    5,   -2,  0.5])
y = np.array([1.65, 0.2, -1.69, 4.7, 7.57, -3.2, 0.65])

mdl = fitlm(x, y)

pretty_print(mdl)

print()
print("------------------------------------------------------")
print("------------- Multiple linear regression -------------")
print("------------------------------------------------------")
print()

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

print()
print("--------------------------------------------------")
print("------------- One sample t-test -------------")
print("--------------------------------------------------")
print()

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


print()
print("--------------------------------------------------")
print("------------- Two sample t-test -------------")
print("--------------------------------------------------")
print()

x = normrnd(2.5, 4, 1, 50).flatten()
y = normrnd(2.5, 4, 1, 50).flatten()

h, p, ci, _stats = ttest2(x, y)

print()
print("h: ", h)
print("p: ", p)
print("ci: ", ci)
pprint.pp(_stats)

x = normrnd(0,   1, 1, 50).flatten()
y = normrnd(2.5, 1, 1, 50).flatten()

h, p, ci, _stats = ttest2(x, y, alpha = 0.03)

print()
print("h: ", h)
print("p: ", p)
print("ci: ", ci)
pprint.pp(_stats)

h, p, ci, _stats = ttest2(x, y, alternative='left')

print()
print("left tail")
print("h: ", h)
print("p: ", p)
print("ci: ", ci)
pprint.pp(_stats)

h, p, ci, _stats = ttest2(x, y, alternative='right')

print()
print("right tail")
print("h: ", h)
print("p: ", p)
print("ci: ", ci)
pprint.pp(_stats)

x = normrnd(2.5, 3, 1, 50).flatten()
y = normrnd(2.5, 4, 1, 50).flatten()

h, p, ci, _stats = ttest2(x, y, equal_var=False)

print()
print("Unqual variances")
print("h: ", h)
print("p: ", p)
print("ci: ", ci)
pprint.pp(_stats)

x = normrnd(0,   2, 1, 50).flatten()
y = normrnd(2.5, 7, 1, 50).flatten()

h, p, ci, _stats = ttest2(x, y, alpha=0.03, equal_var=False)

print()
print("Unqual variances, aplha = 0.03")
print("h: ", h)
print("p: ", p)
print("ci: ", ci)
pprint.pp(_stats)

print()
print("---------------------------------------------------------")
print("--------- Two sample t-test in Facabook dataset ---------")
print("---------------------------------------------------------")
print()

"""
    The dataset was downloaded from:
      https://archive.ics.uci.edu/dataset/368/facebook+metrics
"""

# Read the CSV file
df = pd.read_csv('data/dataset_Facebook.csv', sep=';')

# Split data into paid and not paid
paid_users = df[df['Paid'] == 1]['Lifetime Engaged Users']
not_paid_users = df[df['Paid'] == 0]['Lifetime Engaged Users']

h, p, ci, _stats = ttest2(not_paid_users, paid_users, equal_var=False)

print()
print("t-test of Paid vs not paing Users about the lifetime engagedment.")
print("h: ", h)
print("p: ", p)
print("ci: ", ci)
pprint.pp(_stats)

h, p, ci, _stats = ttest2(not_paid_users, paid_users, 
                          equal_var=False, alpha=0.03)

print()
print("Same but with alpha = 0.03")
print("h: ", h)
print("p: ", p)
print("ci: ", ci)
pprint.pp(_stats)

print()
print("--------------------------------------------------")
print("--------- One-way Anova ---------")
print("--------------------------------------------------")
print()

x1 = [1,      0, -1,    3,   5, -2, 0.5]
x2 = [-1,     2,  4, -0.5,   1,  1,   0]
x3 = [-0.4, 1.2,  0,    3, 2.5, -1,   6]

# concatenate the arrays column-wise
X = np.column_stack((x1, x2, x3))

# This works also if we flat the matrix and give a group array
# X = X.flatten('F')
# groups = np.array(['a', 'a', 'a', 'a', 'a', 'a', 'a', 'b', 'b', 'b', 'b', 'b', 'b', 'b', 'c', 'c', 'c', 'c', 'c', 'c', 'c'])

p_val, table, f_stat = anova1(X, displayopt=False)

print("p_value: ", p_val)
print("f_stat: ", f_stat)
print(table)

print()
print("--------------------------------------------------")
print("--------- Kruskal-Wallis H-test ---------")
print("--------------------------------------------------")
print()

x1 = np.array([1,      0, -1,    3,   5, -2, 0.5])
x2 = np.array([-1,     2,  4, -0.5,   1,  1,   0])
x3 = np.array([-0.4, 1.2,  0,    3, 2.5, -1,   6])

# concatenate the arrays column-wise
X = np.column_stack((x1, x2, x3))

# This works also if we flat the matrix and give a group array
# flatten the array column-wise
# X = X.flatten('F')
# groups = np.array(['a', 'a', 'a', 'a', 'a', 'a', 'a', 'b', 'b', 'b', 'b', 'b', 'b', 'b', 'c', 'c', 'c', 'c', 'c', 'c', 'c'])

p_val, table, h_stat = kruskalwallis(X)

print("p_value: ", p_val)
print("h_stat: ", h_stat)
print(table)
