from ut_plot import *
from os import path

import copy
import matplotlib.pyplot as plt
import numpy as np

def sample(y, y_min):
  y_max = y.max()
  y_old = y.min()
  y = (y_max - y_min) * (y - y_old) / (y_max - y_old) + y_min
  return y

run_file = path.basename(__file__)
data_file = path.join(data_dir, run_file.replace('.py', '.dta'))

data = np.loadtxt(data_file, dtype=np.float32)

start = 50
n_epoch = 6 * 7 + 1
x = np.arange(n_epoch)
stop = start + n_epoch

fig, ax = plt.subplots(1, 1)
fig.set_size_inches(width, height, forward=True)

kwargs = copy.deepcopy(line_kwargs)
kwargs['label'] = '0.2\\%'
kwargs['linestyle'] = linestyles[0]
ax.plot(x, sample(data[start:stop, 0], data[-1, 0]), **kwargs)

kwargs = copy.deepcopy(line_kwargs)
kwargs['label'] = '1\\%'
kwargs['linestyle'] = linestyles[1]
ax.plot(x, sample(data[start:stop, 1], data[-1, 1]), **kwargs)

kwargs = copy.deepcopy(line_kwargs)
kwargs['label'] = '5\\%'
kwargs['linestyle'] = linestyles[2]
ax.plot(x, sample(data[start:stop, 2], data[-1, 2]), **kwargs)

kwargs = copy.deepcopy(line_kwargs)
kwargs['label'] = '20\\%'
kwargs['linestyle'] = linestyles[3]
ax.plot(x, sample(data[start:stop, 3], data[-1, 3]), **kwargs)

ax.legend(loc='upper right', prop={'size': legend_size})

ax.tick_params(axis='both', which='major', labelsize=tick_size)
ax.set_xlabel('Training Epoch', fontsize=label_size)
ax.set_ylabel('MSE', fontsize=label_size)

ax.set_xlim(x.min(), x.max())
ax.set_xticks(np.arange(6, n_epoch, 12))
ax.set_xticklabels(['%d' % (2 * i) for i in range(1, 5)])

eps_file = path.join(fig_dir, run_file.replace('.py', '.eps'))
fig.savefig(eps_file, format='eps', bbox_inches='tight', pad_inches=pad_inches)
