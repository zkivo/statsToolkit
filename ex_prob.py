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
plt.show()
