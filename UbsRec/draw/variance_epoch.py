from ut_plot import *
from os import path

import copy
import matplotlib.pyplot as plt
import numpy as np

handlelength = 1.64 * (1.25 * rcParams['legend.handlelength'])
handletextpad = 0.4 * rcParams['legend.handletextpad']
rc('legend', handlelength=handlelength, handletextpad=handletextpad)

run_file = path.basename(__file__)
data_file = path.join(data_dir, run_file.replace('.py', '.dta'))

data = np.loadtxt(data_file, dtype=np.float32)

n_epoch = 80 + 1
x = np.arange(n_epoch)

fig, ax = plt.subplots(1, 1)
fig.set_size_inches(width, height, forward=True)

kwargs = copy.deepcopy(line_kwargs)
kwargs['label'] = '0'
kwargs['linestyle'] = name_linestyles['solid']
ax.plot(x, data[:n_epoch, 0], **kwargs)

kwargs = copy.deepcopy(line_kwargs)
kwargs['label'] = '1'
# kwargs['marker'] = markers[2]
kwargs['linestyle'] = name_linestyles['dotted']
kwargs['markevery'] = list(np.arange(5, n_epoch, 20))
ax.plot(x, data[:n_epoch, 5], **kwargs)


kwargs = copy.deepcopy(line_kwargs)
kwargs['label'] = '2$^{-6}$'
kwargs['linestyle'] = name_linestyles['densely dashed']
ax.plot(x, data[:n_epoch, 3], **kwargs)

kwargs = copy.deepcopy(line_kwargs)
kwargs['label'] = '2$^{3}$'
# kwargs['marker'] = markers[3]
kwargs['linestyle'] = name_linestyles['densely dashdotted']
kwargs['markevery'] = list(np.arange(10, n_epoch, 20))
ax.plot(x, data[:n_epoch, 6], **kwargs)

kwargs = copy.deepcopy(line_kwargs)
kwargs['label'] = '2$^{-3}$'
kwargs['linestyle'] = name_linestyles['densely dashdotdotted']
kwargs['markevery'] = list(np.arange(0, n_epoch, 20))
ax.plot(x, data[:n_epoch, 4], **kwargs)

kwargs = copy.deepcopy(line_kwargs)
kwargs['label'] = '2$^{6}$'
# kwargs['marker'] = markers[4]
kwargs['linestyle'] = name_linestyles['densely dashdotdotdotted']
kwargs['markevery'] = list(np.arange(15, n_epoch, 20))
ax.plot(x, data[:n_epoch, 7], **kwargs)

kwargs = copy.deepcopy(line_kwargs)
kwargs['label'] = '2$^{-4}$'
kwargs['linestyle'] = linestyles[2]
# ax.plot(x, data[:n_epoch, 2], **kwargs)
kwargs = copy.deepcopy(line_kwargs)
kwargs['label'] = '2$^{-6}$'
kwargs['linestyle'] = linestyles[1]
# ax.plot(x, data[:n_epoch, 1], **kwargs)


ax.legend(bbox_to_anchor=bbox_to_anchor,
          prop={'size': legend_size},
          mode='expand',
          loc=4,
          ncol=3)

ax.tick_params(axis='both', which='major', labelsize=tick_size)
ax.set_xlabel('Training epoch', fontsize=label_size)
ax.set_ylabel('Propensity variance', fontsize=label_size)

ax.set_xlim(x.min(), x.max())
ax.set_xticks(np.arange(0, n_epoch, 20))
ax.set_xticklabels(['%d' % (2 * i) for i in range(5)])
ax.set_yticks([0.008, 0.025])
ax.set_yticklabels(['10$^{-4}$', '10$^{-3}$'])

eps_file = path.join(fig_dir, run_file.replace('.py', '.eps'))
fig.savefig(eps_file, format='eps', bbox_inches='tight', pad_inches=pad_inches)
