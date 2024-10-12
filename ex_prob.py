from MA660E.probability import * 
from MA660E.utils import *
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

# Add labels and title
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('Probability Density')
ax.set_title('Multivariate Normal Distribution')

plt.show()
