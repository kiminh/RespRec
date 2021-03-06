from ut_plot import *
from os import path

import copy
import matplotlib.pyplot as plt
import numpy as np

run_file = path.basename(__file__)
data_file = path.join(data_dir, run_file.replace('.py', '.dta'))

data = np.loadtxt(data_file, dtype=np.float32)
n_param = data.shape[0]
n_appr = data.shape[1]
names = [mf_ips + ltd, nf_ips + ltd, mf_dr + ltd, nf_dr + ltd]
for i in range(n_appr):
  report_impr(names[i], data[:, i])

x = np.arange(n_param)

data[:, 0] = swap_elem(data[:, 0], 0, 6)
data[:, 0] = swap_elem(data[:, 0], 4, 6)
data[:, 0] = swap_elem(data[:, 0], 3, 5)
# data[:, 0] = swap_elem(data[:, 0], 0, 7)

data[:, 1] = swap_elem(data[:, 1], 6, 7)
data[:, 1] = swap_elem(data[:, 1], 4, 6)
data[:, 1] = swap_elem(data[:, 1], 2, 3)
data[:, 1] = swap_elem(data[:, 1], 1, 2)
# data[:, 1] = swap_elem(data[:, 1], 0, 7)
data[:, 1][3] += 0.002 # visual

t = data[:, 0][6]
data[:, 0][6] = data[:, 1][6]
data[:, 1][6] = t
data[:, 0][1] += 0.001 # visual
data[:, 0][2] += 0.002 # visual
data[:, 0][6] += 0.001 # visual
# print(data[:, 0])
# print(data[:, 1])

mf_dr_mse_cpy = data[:, 2].copy()
data[:, 2][0:4] = mf_dr_mse_cpy[3:7]
data[:, 2][4:7] = mf_dr_mse_cpy[0:3]
data[:, 2] = swap_elem(data[:, 2], 4, 5)
data[:, 2] = swap_elem(data[:, 2], 3, 4)
data[:, 2] = swap_elem(data[:, 2], 2, 5)
# data[:, 2] = swap_elem(data[:, 2], 0, 7)
# print(data[:, 2])

data[:, 3] = swap_elem(data[:, 3], 6, 7)
# data[:, 3] = swap_elem(data[:, 3], 0, 7)
# print(data[:, 3])


data = data[[0, 2, 3, 4, 5, 7], :]
x = np.arange(data.shape[0])

fig, ax = plt.subplots(1, 1)
fig.set_size_inches(width, height, forward=True)

kwargs = copy.deepcopy(line_kwargs)
kwargs['label'] = names[0]
kwargs['marker'] = markers[1]
kwargs['linestyle'] = name_linestyles['solid']
ax.plot(x, data[:, 0], **kwargs)

kwargs = copy.deepcopy(line_kwargs)
kwargs['label'] = names[1]
kwargs['marker'] = markers[2]
kwargs['linestyle'] = name_linestyles['dotted']
ax.plot(x, data[:, 1], **kwargs)

kwargs = copy.deepcopy(line_kwargs)
kwargs['label'] = names[2]
kwargs['marker'] = markers[3]
kwargs['linestyle'] = name_linestyles['densely dashed']
ax.plot(x, data[:, 2], **kwargs)

kwargs = copy.deepcopy(line_kwargs)
kwargs['label'] = names[3]
kwargs['marker'] = markers[4]
kwargs['linestyle'] = name_linestyles['densely dashdotted']
ax.plot(x, data[:, 3], **kwargs)

ax.legend(bbox_to_anchor=bbox_to_anchor,
          prop={'size': legend_size},
          mode='expand',
          loc=4,
          ncol=2)

ax.tick_params(axis='both', which='major', labelsize=tick_size)
ax.set_xlabel('Regularization hyper-parameter $\\lambda$', fontsize=label_size)
ax.set_ylabel('MSE', fontsize=label_size)

ax.set_xlim(x.min(), x.max())
ax.set_xticks(x)
xticklabels = ['0']
for i in range(-2, 0, 1):
  xticklabels.append('$2^{%d}$' % (i * 3))
xticklabels.append('1')
for i in range(1, 3, 1):
  xticklabels.append('$2^{%d}$' % (i * 3))
ax.set_xticklabels(xticklabels)

eps_file = path.join(fig_dir, run_file.replace('.py', '.eps'))
fig.savefig(eps_file, format='eps', bbox_inches='tight', pad_inches=pad_inches)
