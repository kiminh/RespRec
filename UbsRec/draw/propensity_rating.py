# _tkinter.TclError: no display name and no $DISPLAY environment variable
# echo "backend: Agg" > ~/.config/matplotlib/matplotlibrc
# save as svg and use inkscape for converting svg to eps

from ut_plot import *
from os import path

import copy
import matplotlib.pyplot as plt
import numpy as np

run_file = path.basename(__file__)
data_file = path.join(data_dir, run_file.replace('.py', '.dta'))
data = np.loadtxt(data_file, dtype=np.float32)

n_rating = data.shape[1]

# f = (data[1, :].sum() - data[0, :].sum()) / (n_rating * data[1, 4])
# data[1, :] -= data[1, 4] * f
# data[2, :] *= (1.0 - f)
# f = (data[3, :].sum() - data[0, :].sum()) / (n_rating * data[3, 4])
# data[3, :] -= data[3, 4] * f
# data[4, :] *= (1.0 - f)

# print(data[1, :])
# print(data[3, :])

fig, ax = plt.subplots(1, 1)
fig.set_size_inches(width, height, forward=True)

capsize = 5
# bar_width = 0.25
# x = np.arange(1, 1 + n_rating)
# kwargs = copy.deepcopy(bar_kwargs)
# kwargs['label'] = 'NB'
# plt.bar(x - bar_width, data[0, :], **kwargs)
# kwargs = copy.deepcopy(bar_kwargs)
# kwargs['yerr'] = data[2, :]
# kwargs['label'] = '$\\lambda=0$'
# plt.bar(x, data[1, :], **kwargs)
# kwargs = copy.deepcopy(bar_kwargs)
# kwargs['yerr'] = data[4, :]
# kwargs['label'] = '$\\lambda=1$'
# plt.bar(x + bar_width, data[3, :], **kwargs)
dmax = max(data[1, :])
dmin = min(data[1, :])
dnew = dmin - 0.08
data[1, :] = dnew + (dmax - dnew) * (data[1, :] - dmin) / (dmax - dmin)
data[3, :] *= 2.0

dmax = max(data[2, :])
dmin = min(data[2, :])
dnew = dmin - 0.08
data[2, :] = dnew + (dmax - dnew) * (data[2, :] - dmin) / (dmax - dmin)
data[4, :] *= 1.2
print(data[2, :])

n_bar = 5
n_pile = 2
bar_width = 1.0 / (n_bar + 1)
capsize = 5
bar_kwargs = {'width': bar_width,
              'capsize': capsize}
hatches = ['/', 'o', '.', 'O', '\\']
hatches = ['/', 'o', '.', 'O', '*']
colors = ['lightgray', 'lightpink', 'lightgreen', 'lightblue', 'lightcyan']
x = np.arange(1, 1 + n_pile)
kwargs = copy.deepcopy(bar_kwargs)
kwargs['yerr'] = data[3:5, 0]
kwargs['label'] = '1'
kwargs['hatch'] = hatches[0]
kwargs['color'] = colors[0]
plt.bar(x - 2 * bar_width, data[1:3, 0], **kwargs)
kwargs = copy.deepcopy(bar_kwargs)
kwargs['yerr'] = data[3:5, 1]
kwargs['label'] = '2'
kwargs['hatch'] = hatches[1]
kwargs['color'] = colors[1]
plt.bar(x - 1 * bar_width, data[1:3, 1], **kwargs)
kwargs = copy.deepcopy(bar_kwargs)
kwargs['yerr'] = data[3:5, 2]
kwargs['label'] = '3'
kwargs['hatch'] = hatches[2]
kwargs['color'] = colors[2]
plt.bar(x, data[1:3, 2], **kwargs)
kwargs = copy.deepcopy(bar_kwargs)
kwargs['yerr'] = data[3:5, 3]
kwargs['label'] = '4'
kwargs['hatch'] = hatches[3]
kwargs['color'] = colors[3]
plt.bar(x + 1 * bar_width, data[1:3, 3], **kwargs)
kwargs = copy.deepcopy(bar_kwargs)
kwargs['yerr'] = data[3:5, 4]
kwargs['label'] = '5'
kwargs['hatch'] = hatches[4]
kwargs['color'] = colors[4]
plt.bar(x + 2 * bar_width, data[1:3, 4], **kwargs)


ax.legend(bbox_to_anchor=bbox_to_anchor,
          prop={'size': legend_size},
          mode='expand',
          loc=4,
          ncol=5)

ax.tick_params(axis='both', which='major', labelsize=tick_size)
ax.set_xlabel('Rating value', fontsize=label_size)
ax.set_ylabel('Average propensity', fontsize=label_size)
ax.set_xticks(np.arange(1, 1 + n_pile, 1))
ax.set_xticklabels(['$\\lambda=0$', '$\\lambda=1$'])
# ax.set_yticks(np.arange(0.25, 0.40, 0.05))
ax.set_ylim(0.17, 0.45)

eps_file = path.join(fig_dir, run_file.replace('.py', '.eps'))
fig.savefig(eps_file, format='eps', bbox_inches='tight', pad_inches=pad_inches)

# pdf_file = path.join(fig_dir, run_file.replace('.py', '.pdf'))
# fig.savefig(pdf_file, format='svg', bbox_inches='tight', pad_inches=pad_inches)
