from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D
import random
import torch
import numpy as np
import h5py
import math
import sys
sys.path.append("..")
from data import ModelNet40
from utils import knn_point, index_points, square_distance
# https://www.semicolonworld.com/question/44454/python-matplotlib-plotting-a-3d-cube-a-sphere-and-a-vector


indexed_point = [-0.6179733,  -0.01154609,  0.13754988]
neighbors=[[-6.0331327e-01, -3.5651371e-02,  8.8275500e-02],
 [-5.8656329e-01, -2.5616191e-02,  1.8500750e-01],
 [-6.4876240e-01,  6.9144275e-03,  8.4561124e-02],
 [-6.8672258e-01, -2.3941665e-03,  1.5412858e-01],
 [-6.0497177e-01,  1.2594928e-03,  6.5689333e-02],
 [-5.4834837e-01, -1.3680734e-02,  9.6334979e-02],
 [-7.0411325e-01,  1.2344187e-03,  1.1237644e-01],
 [-6.0628176e-01, -6.4614125e-02,  4.7462806e-02],
 [-6.5674597e-01,  1.1000191e-02,  3.0012082e-02],
 [-5.0481933e-01, -4.2370517e-02,  1.2853034e-01],
 [-5.7215101e-01, -4.2567228e-04,  2.7258655e-02],
 [-7.1667945e-01,  3.1107110e-03,  2.0460460e-01],
 [-5.9712839e-01, -3.5986330e-02,  1.2272621e-02],
 [-5.0256950e-01, -4.6524893e-02,  7.6449126e-02],
 [-6.2563801e-01,  7.4670664e-03,  3.3674985e-03],
 [-7.5284290e-01,  6.9724754e-03,  1.5031154e-01],
 [-7.2546268e-01,  1.4036452e-02,  5.0331891e-02],
 [-7.6129931e-01,  9.8874085e-03,  9.7600281e-02],
 [-5.2005595e-01, -8.3800135e-03,  1.5476015e-02]]


abs_neighbors = neighbors - np.expand_dims(indexed_point, axis=0)
abs_neighbors = np.array(abs_neighbors)
scalar = 6
abs_neighbors = abs_neighbors *scalar
print(abs_neighbors.shape)




from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from itertools import product, combinations


fig = plt.figure()
ax = fig.gca(projection='3d')
# ax.set_aspect("equal")

# # draw cube
# r = [-1, 1]
# for s, e in combinations(np.array(list(product(r, r, r))), 2):
#     if np.sum(np.abs(s-e)) == r[1]-r[0]:
#         ax.plot3D(*zip(s, e), color="b")

# draw sphere
u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
x = np.cos(u)*np.sin(v)
y = np.sin(u)*np.sin(v)
z = np.cos(v)
ax.plot_wireframe(x, y, z, color="lightgray", linewidth=0.5, linestyle='dashed')

# draw a point
# ax.scatter([0], [0], [0], color="g", s=100)

# draw a vector
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d


class Arrow3D(FancyArrowPatch):

    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        FancyArrowPatch.draw(self, renderer)

a = Arrow3D([-1.4, 1.4], [0, 0], [0, 0], mutation_scale=20,
            lw=1, arrowstyle="-|>", color="k")
ax.add_artist(a)

b = Arrow3D([0, 0], [-1.4, 1.4], [0, 0], mutation_scale=20,
            lw=1, arrowstyle="-|>", color="k")
ax.add_artist(b)
c = Arrow3D([0, 0], [0,0], [-1.4, 1.4], mutation_scale=20,
            lw=1, arrowstyle="-|>", color="k")
ax.add_artist(c)

# make the panes transparent
ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
# make the grid lines transparent
ax.xaxis._axinfo["grid"]['color'] =  (1,1,1,0)
ax.yaxis._axinfo["grid"]['color'] =  (1,1,1,0)
ax.zaxis._axinfo["grid"]['color'] =  (1,1,1,0)
ax.set_axis_off()
ax.get_xaxis().get_major_formatter().set_useOffset(False)

ax.scatter(0, 0, 0, color = "red", s=80, marker="*")
ax.scatter(abs_neighbors[:, 0], abs_neighbors[:, 1], abs_neighbors[:, 2], color = "limegreen", s=30)
plt.show()
fig.savefig("3d_ball.pdf", bbox_inches='tight', pad_inches=-0.6, transparent=True)


