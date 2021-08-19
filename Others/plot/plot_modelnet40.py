from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D
import random
import numpy as np
import h5py
import math
import sys
sys.path.append("..")
from data import ModelNet40

id = np.random.randint(0,2048)
id=800 #airplane 11
id=2001 # lighter
id=860
color='lightskyblue'
color='yellowgreen'
color='orange'
points=50
save_fig=True
rotation=True
scale=True


datset = ModelNet40(points, partition='test')
sample,label = datset.__getitem__(id)




fig = pyplot.figure()
ax = Axes3D(fig)

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


ax.scatter(sequence_containing_x_vals, sequence_containing_y_vals, sequence_containing_z_vals, color = color)


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
    fig.savefig(f"{id}_{points}.pdf", bbox_inches='tight', pad_inches=0.05, transparent=True)
