from ut_plot import *
from os import path

import copy
import matplotlib.pyplot as plt
import numpy as np

run_file = path.basename(__file__)
data_file = path.join(data_dir, run_file.replace('.py', '.dta'))

data = np.loadtxt(data_file, dtype=np.float32)

n_epoch = data.shape[0]
x = np.arange(n_epoch)

fig, ax = plt.subplots(1, 1)
fig.set_size_inches(width, height, forward=True)

n_kwargs = copy.deepcopy(c_kwargs)
n_kwargs['label'] = '0'
n_kwargs['linestyle'] = linestyles[0]
ax.plot(x, data[:, 0], **n_kwargs)

n_kwargs = copy.deepcopy(c_kwargs)
n_kwargs['label'] = '10$^{-3}$'
n_kwargs['linestyle'] = linestyles[1]
ax.plot(x, data[:, 1], **n_kwargs)

n_kwargs = copy.deepcopy(c_kwargs)
n_kwargs['label'] = '10$^{-2}$'
n_kwargs['linestyle'] = linestyles[2]
ax.plot(x, data[:, 2], **n_kwargs)

n_kwargs = copy.deepcopy(c_kwargs)
n_kwargs['label'] = '10$^{-1}$'
n_kwargs['linestyle'] = linestyles[3]
ax.plot(x, data[:, 3], **n_kwargs)

n_kwargs = copy.deepcopy(c_kwargs)
n_kwargs['label'] = '10$^{0}$'
n_kwargs['linestyle'] = linestyles[0]
n_kwargs['marker'] = markers[0]
n_kwargs['markevery'] = list(np.arange(0, n_epoch, 20))
ax.plot(x, data[:, 4], **n_kwargs)

n_kwargs = copy.deepcopy(c_kwargs)
n_kwargs['label'] = '10$^{1}$'
n_kwargs['linestyle'] = linestyles[1]
n_kwargs['marker'] = markers[1]
n_kwargs['markevery'] = list(np.arange(5, n_epoch, 20))
ax.plot(x, data[:, 5], **n_kwargs)

n_kwargs = copy.deepcopy(c_kwargs)
n_kwargs['label'] = '10$^{2}$'
n_kwargs['linestyle'] = linestyles[2]
n_kwargs['marker'] = markers[2]
n_kwargs['markevery'] = list(np.arange(10, n_epoch, 20))
ax.plot(x, data[:, 6], **n_kwargs)

n_kwargs = copy.deepcopy(c_kwargs)
n_kwargs['label'] = '10$^{3}$'
n_kwargs['linestyle'] = linestyles[3]
n_kwargs['marker'] = markers[3]
n_kwargs['markevery'] = list(np.arange(15, n_epoch, 20))
ax.plot(x, data[:, 7], **n_kwargs)

ax.legend(bbox_to_anchor=bbox_to_anchor,
          prop={'size': legend_size},
          mode='expand',
          loc=4,
          ncol=4)

eps_file = path.join(fig_dir, run_file.replace('.py', '.eps'))
fig.savefig(eps_file, format='eps', bbox_inches='tight', pad_inches=pad_inches)
