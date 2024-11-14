from statstoolkit.probability import * 
from statstoolkit.utils import *
import matplotlib.pyplot as plt

print()
print("-----------------------------------------------")
print("------------- Normal Distribution -------------")
print("-----------------------------------------------")
print()

mu = -1.5
sigma = 2

x = mu + linspace(-5.5, 5.5, 100)  # Create x values
y = normpdf(x, mu, sigma)  # Compute the probability density function

plt.figure()
plt.plot(x, y)
plt.axis([-6, 3, 0, 0.25])  # Set the axis limits
plt.axhline(0, color='black', linestyle='dashdot', linewidth=0.5)  # x-axis
plt.axvline(0, color='black', linestyle='dashdot', linewidth=0.5)  # y-axis
plt.title('PDF of N(-1.5, 2)')

print()
print("------------------------------------------------------------")
print("------------- Multivariate Normal Distribution -------------")
print("------------------------------------------------------------")
print()

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

print()
print("-------------------------------------------------")
print("------------- Generate a Custom PDF -------------")
print("-------------------------------------------------")
print()

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

print()
print("-------------------------------------------------")
print("------------- Hypergeometric Distribution -------------")
print("-------------------------------------------------")
print()

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

print()
print("--------------------------------------")
print("------------- Custom CDF -------------")
print("--------------------------------------")
print()

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

print()
print("-------------------------------------------------")
print("------------- Binomial CDF examples -------------")
print("-------------------------------------------------")
print()

print(binocdf(20, 75, 0.23)) # 0.8153
print(1 - binocdf(14, 75, 0.23)) # 0.7714
print(binocdf(21, 75, 0.23) - binocdf(6, 75, 0.23)) #0.8762

print(tcdf(2.31, 73)) # 0.9881
print(1 - tcdf(8.25, 73)) # 2.3865e-12
print(tcdf(9.63, 73) - tcdf(3.17, 73)) # 0.0011

print()
print("---------------------------------------------------------------------")
print("------------- Calculate the probability of a custom PDF -------------")
print("---------------------------------------------------------------------")
print()

def denf(x, c):
    return c * (1 + x) * (2 + np.sin(3 * x))

c0 = 1 / (24 + 1 / 3 + (np.sin(12) - 15 * np.cos(12)) / 9)

Prob = integral(lambda x: denf(x, c0), 1, 3)

print("Prob:", Prob) # 0.5503

# second example

c = 1 / (1 / 6 + 3 / 16 + 1 / (7 * 2 ** 7))

def denf_2d(y, x):
    return c * (y ** 2 + 3 * x * y + 4 * y ** 5 * x ** 6)

def ymin(x):
    return x ** (1 / 2)

def ymax(x):
    return x ** (1 / 3)

prob_2d = integral2(denf_2d, 0, 1 / 2, ymin, ymax)

print("prob_2d:", prob_2d) # 0.1246

print()
print("--------------------------------------------------------------")
print("------------- Calculate Expectation and Variance -------------")
print("--------------------------------------------------------------")
print()

def denf(x, c):
    return c * x * (1 + x) * (2 + np.sin(3 * x))

def denvf(x, c):
    return c * (x - Mu_x) ** 2 * (1 + x) * (2 + np.sin(3 * x))

c0 = 1 / (24 + 1 / 3 + (np.sin(12) - 15 * np.cos(12)) / 9)

Mu_x = integral(lambda x: denf(x, c0), 1, 3)
V_x  = integral(lambda x: denvf(x, c0), 1, 3)

print("Mu_x (mean):", Mu_x) # 1.2506
print("V_x (variance):", V_x) # 0.7344

print()
print("-------------------------------------------")
print("------------- Critical values -------------")
print("-------------------------------------------")
print()

alpha = 0.0132
print(norminv(1 - alpha)) # 2.2203
print(norminv(1 - alpha / 2)) # 2.4783

alpha = 0.012
v = 61
print(tinv(1 - alpha, v)) # 2.3149
print(tinv(1 - alpha / 2, v)) # 2.5896

alpha = 0.012
v = 19
print(chi2inv(1 - alpha, v)) # 35.5444
print(chi2inv(1 - alpha / 2, v)) # 37.9626
print(chi2inv(alpha / 2, v)) # 7.0394

alpha = 0.012
v1 = 9
v2 = 13
print(finv(1 - alpha, v1, v2)) # 4.0072
print(finv(1 - alpha / 2, v1, v2)) # 4.7321
print(finv(alpha / 2, v1, v2)) # 0.1710

print()
print("--------------------------------------------------")
print("------------- Generate random values -------------")
print("--------------------------------------------------")
print()


print(randperm(7))              # [2 1 5 4 6 3 7]
print(randi([1, 350], 1, 5))    # [[  8 292 240 296 196]]
print(randi([1, 350], 6))       # [ 45  32 131 346 314 300]
print(rand(1,5))                # [[0.73112933 0.5274633  0.25467238 0.1383535  0.37195866]]
print(-3+5*rand(1, 6))          # [[-1.21751906  0.53154677  0.38587971  0.87498599  1.5511747   1.6080525 ]]
print(normrnd(-1, 0.5, 1, 6))   # [[-0.9086999  -0.94865145 -1.18145306 -0.93692477 -0.36856358 -1.25120495]]
print(chi2rnd(4, 1, 4))         # [[5.04566035 5.48997902 0.86619863 1.36271316]]

plt.show()
