import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
norm.cdf(1.96)


def solve(m1, m2, std1, std2):
    a = 1 / (2 * std1**2) - 1 / (2 * std2**2)
    b = m2 / (std2**2) - m1 / (std1**2)
    c = m1**2 / (2 * std1**2) - m2**2 / (2 * std2**2) - np.log(std2 / std1)
    return np.roots([a, b, c])


m1 = 2.5
std1 = 0.5
m2 = 5.0
std2 = 0.1

area_list = []
# Get point of intersect
# for i in np.arange(10, 3, -0.1):
#     result = solve(m1, i, std1, std2)
#     r = result[0]
#     area = norm.cdf(r, m2, std2) + (1. - norm.cdf(r, m1, std1))
#     area_list.append(area)
#     print area

# Get point on surface
result = solve(m1, m2, std1, std2)
x = np.linspace(-5, 9, 10000)
plot1 = plt.plot(x, norm.pdf(x, m1, std1))
plot2 = plt.plot(x, norm.pdf(x, m2, std2))
plot3 = plt.plot(result, norm.pdf(result, m1, std1), 'o')

# Plots integrated area
r = result[0]
olap = plt.fill_between(x[x > r], 0, norm.pdf(x[x > r], m1, std1), alpha=0.3)
olap = plt.fill_between(x[x < r], 0, norm.pdf(x[x < r], m2, std2), alpha=0.3)

# integrate
area = norm.cdf(r, m2, std2) + (1. - norm.cdf(r, m1, std1))
print("Area under curves ", area)
# plt.plot(np.arange(10, 3, -0.1), area_list)
plt.show()
