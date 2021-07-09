import numpy as np
import matplotlib.pyplot as plt

cmap = plt.get_cmap('viridis')
names = ["bob", "joe", "andrew", "pete","232"]
colors = cmap(np.linspace(0, 1, len(names)))
import matplotlib.colors as mcolors
def_colors = mcolors.CSS4_COLORS
colrs_list = []
for k,v in def_colors.items():
    print(k)
    colrs_list.append(k)
np.random.shuffle(colrs_list)

print(colors)
print(colors.shape)
# [[ 0.267004  0.004874  0.329415  1.      ]
#  [ 0.190631  0.407061  0.556089  1.      ]
#  [ 0.20803   0.718701  0.472873  1.      ]
#  [ 0.993248  0.906157  0.143936  1.      ]]

x = np.linspace(0, np.pi*2, 100)
for i, (name, color) in enumerate(zip(names, colors), 1):
    color = colors[i-1]
    plt.plot(x, np.sin(x)/i, label=name, c=colrs_list[i])
plt.legend()
plt.show()
