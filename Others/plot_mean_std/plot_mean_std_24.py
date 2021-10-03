import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
# plt.style.use('science')
plt.rcParams["font.family"] = "Times New Roman"
import numpy as np
import os

fig, ax = plt.subplots()
fig.set_size_inches(6, 4)
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



csfont = {'fontname':'Times New Roman'}
x = np.arange(len(pointMLP24noBN_mean))

plt.plot(x, pointMLP24_mean, '-', color='C1', label='w/ Affine',linewidth=0.8)
plt.fill_between(x, pointMLP24_mean - pointMLP24_std, pointMLP24_mean + pointMLP24_std, color='C1', alpha=0.4, linewidth=0.5)

plt.plot(x, pointMLP24noBN_mean, '--', color='C1', label='w/o Affine',linewidth=0.8)
plt.fill_between(x, pointMLP24noBN_mean - pointMLP24noBN_std, pointMLP24noBN_mean + pointMLP24noBN_std, color='C1', alpha=0.2, linewidth=0.1)


plt.ylabel('Overall accuracy (OA)', fontsize=18)
plt.xlabel('Training epoch', fontsize=18)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.ylim([30,88])
plt.legend(fontsize=16)


# Make the zoom-in plot:
x1 = 50
x2 = 75
y1 = 75
y2 = 85
# axins = zoomed_inset_axes(ax, 2, loc=8) # zoom = 2
axins = zoomed_inset_axes(ax, 3.2, bbox_to_anchor=[250,190]) # zoom = 2
# axins.plot(pointMLP40)
axins.plot(x, pointMLP24_mean, '-',color='C1', linewidth=0.8)
axins.fill_between(x, pointMLP24_mean - pointMLP24_std, pointMLP24_mean + pointMLP24_std, color='C1', alpha=0.4, linewidth=0.5)
axins.plot(x, pointMLP24noBN_mean, '--',color='C1', linewidth=0.8)
axins.fill_between(x, pointMLP24noBN_mean - pointMLP24noBN_std, pointMLP24noBN_mean + pointMLP24noBN_std, color='C1', alpha=0.2, linewidth=0.1)

axins.set_xlim(x1, x2)
axins.set_ylim(y1, y2)
plt.xticks(visible=False)
plt.yticks(visible=False)
mark_inset(ax, axins, loc1=1, loc2=2, fc="none", ec="0.5")

plt.show()
fig.savefig("with_without_affine_24.pdf", bbox_inches='tight', transparent=True)


