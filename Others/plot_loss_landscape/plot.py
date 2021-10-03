import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import numpy as np
from numpy import genfromtxt
import torch
import torch.nn.functional as F
plt.rcParams["font.family"] = "Times New Roman"

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

# Make data.
filename="model31C_loss_landscape"
# filename="pointnet2_loss_landscape"
# filename="model31CNoRes_loss_landscape"
data = genfromtxt(filename+'.txt', delimiter=',')
where_are_NaNs = np.isnan(data)
data[where_are_NaNs] = 10000000
data = np.clip(data, a_min=-99999, a_max=100000000)


X = np.arange(-1, 1.1, 0.01)
Y = np.arange(-1, 1.1, 0.01)
X, Y = np.meshgrid(X, Y)

Z = data[:,2]  # [dirct1, direct2, loss, acc]
Z = Z.reshape(21,21)
Z = torch.from_numpy(Z).unsqueeze(dim=0).unsqueeze(dim=0)
ZZ = F.interpolate(Z, scale_factor=10, mode="bicubic").squeeze(dim=0).squeeze(dim=0).numpy()
# print(Z.size)

# Plot the surface.
surf = ax.plot_surface(X, Y, ZZ, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

# Customize the z axis.
ax.set_zlim(1.5, 4)
ax.grid(linestyle='dashed')
# ax.zaxis.set_major_locator(LinearLocator(10))
# A StrMethodFormatter is used automatically
# ax.zaxis.set_major_formatter('{x:.02f}')

# Add a color bar which maps values to colors.
# fig.colorbar(surf, shrink=0.5, aspect=5)

ax.get_xaxis().set_visible(True)
ax.axes.get_yaxis().set_visible(True)
ax.set_zlabel('Testing loss', fontsize=16)
# plt.axis('off')
plt.show()
fig.savefig(f"{filename}.pdf", bbox_inches='tight', pad_inches=0, transparent=True)
