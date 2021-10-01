import matplotlib.pyplot as plt
import numpy as np
import os

path = os.getcwd()
filename = "model31C.txt"
data = np.loadtxt(os.path.join(path,filename), dtype=float)
mean_1 = np.array([10, 20, 30, 25, 32, 43])
std_1 = np.array([2.2, 2.3, 1.2, 2.2, 1.8, 3.5])

mean_2 = np.array([12, 22, 30, 13, 33, 39])
std_2 = np.array([2.4, 1.3, 2.2, 1.2, 1.9, 3.5])

x = np.arange(len(mean_1))
plt.plot(x, mean_1, 'b-', label='mean_1')
plt.fill_between(x, mean_1 - std_1, mean_1 + std_1, color='b', alpha=0.2)
plt.plot(x, mean_2, 'r-', label='mean_2')
plt.fill_between(x, mean_2 - std_2, mean_2 + std_2, color='r', alpha=0.2)
plt.legend()
plt.show()
