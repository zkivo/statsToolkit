from statstoolkit.probability import * 
from statstoolkit.utils import *
import matplotlib.pyplot as plt

# ----------------------------- #
# Example 1: Normal Distribution
# ----------------------------- #

mu = -1.5
sigma = 2

x = mu + linspace(-5.5, 5.5, 100)  # Create x values
y = normpdf(x, mu, sigma)  # Compute the probability density function

plt.figure()
plt.plot(x, y)
plt.axis([-6, 3, 0, 0.25])  # Set the axis limits
plt.title('PDF of N(-1.5, 2)')

# ----------------------------- #
# Example 2: Multivariate Normal Distribution
# ----------------------------- #

# Define the mean (mu) and covariance matrix (sigma)
mu = np.array([-1.5, 2])
sigma = np.array([[1.5, -0.4], [-0.4, 2]])

# Define the x and y ranges
x = mu[0] + linspace(-5, 5, 100)
y = mu[1] + linspace(-5, 5, 100)

# Create a meshgrid
X, Y = meshgrid(x, y)
W = np.column_stack([X.ravel(), Y.ravel()])

# Compute the multivariate normal probability density function
Z = mvnpdf(W, mean=mu, cov=sigma)

# Reshape Z to match the shape of the meshgrid
Z = Z.reshape(X.shape)

# Plot the surface
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap='viridis')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('Probability Density')
ax.set_title('Multivariate Normal Distribution')

# ----------------------------- #
# Example 3: Generate a Custom PDF
# ----------------------------- #

# Define the PDF function
def genpdf(x):
    return 0.5 * (1 + x**3)

# Generate values for x
x = linspace(-1, 1, 100)
y = genpdf(x)

# Plot the PDF
plt.figure()
plt.plot(x, y, 'b', label='0.5 * (1 + x^3)')
plt.plot(0.5 * x - 1.5, 0 * x, 'b-.')
plt.plot(0.5 * x + 1.5, 0 * x, 'b-.')
plt.axhline(0, color='black',linewidth=0.5)  # x-axis
plt.axvline(0, color='black',linewidth=0.5)  # y-axis
plt.xlabel('x')
plt.ylabel('Probability Density')
plt.legend()
# plt.grid(True)

# ----------------------------- #
# Example 4: Hypergeometric Distribution
# ----------------------------- #

# Parameters
M = 500  # Population size
K = 30   # Number of successes in population
n = 15   # Sample size

# Generate values for x
x = np.arange(0, 15)
y = hygecdf(x, M, K, n)

# Plot the CDF
plt.figure()
plt.plot(x, y, 'ro')
plt.step(x, y)  # Stairs plot (commented out similar to the MATLAB code)
plt.title('CDF of Hypergeometric')
plt.grid(True)

# ----------------------------- #
# Example 5: Custom CDF
# ----------------------------- #

# Define the function gencf
def gencf(x):
    return 0.375 + 0.5 * x + 0.125 * x**4

# Generate values for x
x = linspace(-1, 1, 100)
y = gencf(x)

# Create the plot
plt.figure()
plt.plot(x, y, 'b', label='0.375 + 0.5*x + 0.125*x^4')
plt.plot(0.5 * x - 1.5, 0 * x, 'b-.')
plt.plot(0.5 * x + 1.5, (0 * x) + 1, 'b-.')
plt.xlabel('x')
plt.ylabel('Probability Density')
plt.legend()
plt.grid(True)


plt.show()
