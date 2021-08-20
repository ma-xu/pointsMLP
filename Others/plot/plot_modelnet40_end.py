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
import matplotlib.pyplot as plt


id=21
color='orange'
points=80 # dont change the id and points, the point_id will not look good
save_fig=True
rotation=True
scale=True

# point_id = 381  # 459, 470, 240 tail ,310, 330
# k_neighbors=20


datset = ModelNet40(points, partition='test')
sample,label = datset.__getitem__(id)

# indexed_point = sample[point_id]
# indexed_point_torch = torch.Tensor(indexed_point).view(1,1,3)
# sample_torch = torch.Tensor(sample).unsqueeze(dim=0)
# idx = knn_point(k_neighbors, sample_torch, indexed_point_torch)
# neighbor_points = index_points(sample_torch, idx[:,:,1:]).squeeze(dim=0).squeeze(dim=0).numpy()



fig = pyplot.figure()
ax = Axes3D(fig)


# sample = np.delete(sample, idx, 0)
sequence_containing_x_vals = sample[:, 0]
x_min = min(sequence_containing_x_vals)
x_max = max(sequence_containing_x_vals)
print(f"x range: {x_max-x_min}")
sequence_containing_y_vals = sample[:, 1]
y_min = min(sequence_containing_y_vals)
y_max = max(sequence_containing_y_vals)
print(f"y range: {y_max-y_min}")
sequence_containing_z_vals = sample[:, 2]
z_min = min(sequence_containing_z_vals)
z_max = max(sequence_containing_z_vals)
print(f"z range: {z_max-z_min}")

# cmap = plt.get_cmap('prism')
# colors = cmap(np.linspace(0, 1, len(sequence_containing_x_vals)))
# ax.scatter(sequence_containing_x_vals, sequence_containing_y_vals, sequence_containing_z_vals, color = colors, s=110)

colors = 5*sequence_containing_x_vals+3*sequence_containing_y_vals+2*sequence_containing_z_vals
norm = pyplot.Normalize(vmin=min(colors), vmax=max(colors))
ax.scatter(sequence_containing_x_vals, sequence_containing_y_vals, sequence_containing_z_vals,
           c=colors, s=110, cmap='rainbow', norm=norm)

# add indexed point
# ax.scatter(indexed_point[0], indexed_point[1], indexed_point[2], color = "red", s=80, marker="*")
# ax.scatter(neighbor_points[:, 0], neighbor_points[:, 1], neighbor_points[:, 2], color = "limegreen", s=30)


# make the panes transparent
ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
# make the grid lines transparent
ax.xaxis._axinfo["grid"]['color'] =  (1,1,1,0)
ax.yaxis._axinfo["grid"]['color'] =  (1,1,1,0)
ax.zaxis._axinfo["grid"]['color'] =  (1,1,1,0)
# Make panes transparent
ax.set_xlim3d(x_min,x_max)
ax.set_ylim3d(y_min,y_max)
ax.set_zlim3d(z_min,z_max)

ax.set_axis_off()
ax.get_xaxis().get_major_formatter().set_useOffset(False)
# pyplot.tight_layout()
pyplot.show()
if save_fig:
    fig.tight_layout()
    # ax = fig.add_axes([0.6, 0.6, 0.6, 0.6])
    fig.savefig(f"{id}_{points}_end.pdf", bbox_inches='tight', pad_inches=-0.7, transparent=True)
