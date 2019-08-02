from ut_plot import *
from os import path

import copy
import matplotlib.pyplot as plt
import numpy as np

handletextpad = 0.4 * rcParams['legend.handletextpad']
rc('legend', handlelength=handlelength, handletextpad=handletextpad)

run_file = path.basename(__file__)
data_file = path.join(data_dir, run_file.replace('.py', '.dta'))
data = np.loadtxt(data_file, dtype=np.float32)
scale = data[-1, :] / data[-2, :]
dnew = data[-1, :]
dmin = data[-2, :]
dmax = data[0, :]
data = dnew + (dmax - dnew) * (data - dmin) / (dmax - dmin)
data = data[:-1, :]
print(data)

n_epoch = 80 + 1
x = np.arange(n_epoch)

fig, ax = plt.subplots(1, 1)
fig.set_size_inches(width, height, forward=True)

kwargs = copy.deepcopy(line_kwargs)
kwargs['label'] = '1'
# kwargs['marker'] = markers[0]
kwargs['linestyle'] = name_linestyles['solid']
kwargs['markevery'] = list(np.arange(16, n_epoch, 20))
ax.plot(x, data[:n_epoch, 0], **kwargs)

kwargs = copy.deepcopy(line_kwargs)
kwargs['label'] = '2'
# kwargs['marker'] = markers[1]
kwargs['linestyle'] = name_linestyles['dotted']
kwargs['markevery'] = list(np.arange(12, n_epoch, 20))
ax.plot(x, data[:n_epoch, 1], **kwargs)

kwargs = copy.deepcopy(line_kwargs)
kwargs['label'] = '3'
# kwargs['marker'] = markers[2]
kwargs['linestyle'] = name_linestyles['densely dashed']
kwargs['markevery'] = list(np.arange(8, n_epoch, 20))
ax.plot(x, data[:n_epoch, 2], **kwargs)

kwargs = copy.deepcopy(line_kwargs)
kwargs['label'] = '4'
# kwargs['marker'] = markers[3]
kwargs['linestyle'] = name_linestyles['densely dashdotted']
kwargs['markevery'] = list(np.arange(4, n_epoch, 20))
ax.plot(x, data[:n_epoch, 3], **kwargs)

kwargs = copy.deepcopy(line_kwargs)
kwargs['label'] = '5'
# kwargs['marker'] = markers[4]
kwargs['linestyle'] = name_linestyles['densely dashdotdotted']
kwargs['markevery'] = list(np.arange(0, n_epoch, 20))
ax.plot(x, data[:n_epoch, 4], **kwargs)

ax.legend(bbox_to_anchor=bbox_to_anchor,
          prop={'size': legend_size},
          mode='expand',
          loc=4,
          ncol=5)

ax.tick_params(axis='both', which='major', labelsize=tick_size)
ax.set_xlabel('Training Epoch', fontsize=label_size)
ax.set_ylabel('Average Propensity', fontsize=label_size)

ax.set_xlim(x.min(), x.max())
ax.set_xticks(np.arange(0, n_epoch, 20))
ax.set_xticklabels(['%d' % (2 * i) for i in range(5)])

eps_file = path.join(fig_dir, run_file.replace('.py', '.eps'))
fig.savefig(eps_file, format='eps', bbox_inches='tight', pad_inches=pad_inches)
