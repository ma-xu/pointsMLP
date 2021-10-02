import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
# plt.style.use('science')
plt.rcParams["font.family"] = "Times New Roman"
import numpy as np
import os

fig, ax = plt.subplots()
fig.set_size_inches(7, 5)
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
ax.grid(linestyle='dashed')

path = os.getcwd()



filename = "pointMLP24.txt"
pointMLP24 = np.loadtxt(os.path.join(path,filename), dtype=float)
pointMLP24_mean = pointMLP24.mean(axis=1)
pointMLP24_std = pointMLP24.std(axis=1)


filename = "pointMLP24_noBN.txt"
pointMLP24noBN = np.loadtxt(os.path.join(path,filename), dtype=float)
pointMLP24noBN_mean = pointMLP24noBN.mean(axis=1)
pointMLP24noBN_std = pointMLP24noBN.std(axis=1)

filename = "pointMLP40.txt"
pointMLP40 = np.loadtxt(os.path.join(path,filename), dtype=float)
pointMLP40_mean = pointMLP40.mean(axis=1)
pointMLP40_std = pointMLP40.std(axis=1)

filename = "pointMLP40_noBN.txt"
pointMLP40noBN = np.loadtxt(os.path.join(path,filename), dtype=float)
pointMLP40noBN_mean = pointMLP40noBN.mean(axis=1)
pointMLP40noBN_std = pointMLP40noBN.std(axis=1)

filename = "pointMLP56.txt"
pointMLP56 = np.loadtxt(os.path.join(path,filename), dtype=float)
pointMLP56_mean = pointMLP56.mean(axis=1)
pointMLP56_std = pointMLP56.std(axis=1)


filename = "pointMLP56_noBN.txt"
pointMLP56noBN = np.loadtxt(os.path.join(path,filename), dtype=float)
pointMLP56noBN_mean = pointMLP56noBN.mean(axis=1)
pointMLP56noBN_std = pointMLP56noBN.std(axis=1)


csfont = {'fontname':'Times New Roman'}
x = np.arange(len(pointMLP40_mean))

plt.plot(x, pointMLP24_mean, 'c-', label='24-Layers w/ Affine',linewidth=0.8)
plt.fill_between(x, pointMLP24_mean - pointMLP24_std, pointMLP24_mean + pointMLP24_std, color='c', alpha=0.4, linewidth=0.5)

plt.plot(x, pointMLP24noBN_mean, 'c--', label='24-Layers w/o Affine',linewidth=0.8)
plt.fill_between(x, pointMLP24noBN_mean - pointMLP24noBN_std, pointMLP24noBN_mean + pointMLP24noBN_std, color='c', alpha=0.2, linewidth=0.1)



plt.plot(x, pointMLP40_mean, 'm-', label='40-Layers w/ Affine',linewidth=0.8)
plt.fill_between(x, pointMLP40_mean - pointMLP40_std, pointMLP40_mean + pointMLP40_std, color='m', alpha=0.4, linewidth=0.5)

plt.plot(x, pointMLP40noBN_mean, 'm--', label='40-Layers w/o Affine',linewidth=0.8)
plt.fill_between(x, pointMLP40noBN_mean - pointMLP40noBN_std, pointMLP40noBN_mean + pointMLP40noBN_std, color='m', alpha=0.4, linewidth=0.5)


plt.plot(x, pointMLP56_mean, 'y-', label='56-Layers w/ Affine',linewidth=0.8)
plt.fill_between(x, pointMLP56_mean - pointMLP56_std, pointMLP56_mean + pointMLP56_std, color='y', alpha=0.4, linewidth=0.5)

plt.plot(x, pointMLP56noBN_mean, 'y--', label='56-Layers w/o Affine',linewidth=0.8)
plt.fill_between(x, pointMLP56noBN_mean - pointMLP56noBN_std, pointMLP56noBN_mean + pointMLP56noBN_std, color='y', alpha=0.2, linewidth=0.1)


plt.ylabel('Overall accuracy (OA)', fontsize=18)
plt.xlabel('Training epoch', fontsize=18)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.ylim([30,88])
plt.legend(fontsize=16)


# Make the zoom-in plot:
x1 = 175
x2 = 200
y1 = 78.5
y2 = 86
# axins = zoomed_inset_axes(ax, 2, loc=8) # zoom = 2
axins = zoomed_inset_axes(ax, 3, bbox_to_anchor=[250,200]) # zoom = 2
# axins.plot(pointMLP40)
axins.plot(x, pointMLP40_mean, 'm-', label='40-Layer w/ Affine',linewidth=0.8)
axins.fill_between(x, pointMLP40_mean - pointMLP40_std, pointMLP40_mean + pointMLP40_std, color='m', alpha=0.4, linewidth=0.5)
axins.plot(x, pointMLP40noBN_mean, 'm--', label='40-Layers w/o Affine',linewidth=0.8)
axins.fill_between(x, pointMLP40noBN_mean - pointMLP40noBN_std, pointMLP40noBN_mean + pointMLP40noBN_std, color='m', alpha=0.2, linewidth=0.1)

axins.set_xlim(x1, x2)
axins.set_ylim(y1, y2)
plt.xticks(visible=False)
plt.yticks(visible=False)
mark_inset(ax, axins, loc1=1, loc2=2, fc="none", ec="0.5")

plt.show()
fig.savefig("with_without_affine.pdf", bbox_inches='tight', transparent=True)


