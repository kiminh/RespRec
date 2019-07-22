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
data = np.flip(data, axis=0)

x = np.arange(data.shape[0])

fig, ax = plt.subplots(1, 1)
fig.set_size_inches(width, height, forward=True)

kwargs = copy.deepcopy(line_kwargs)
kwargs['label'] = mf_ips + ltd
kwargs['label'] = '%s/%s' % (mf_dr , 'MF')
kwargs['marker'] = markers[1]
kwargs['linestyle'] = linestyles[0]
ax.plot(x, data[:, 0], **kwargs)

kwargs = copy.deepcopy(line_kwargs)
kwargs['label'] = nf_ips + ltd
kwargs['label'] = '%s/%s' % (mf_dr + ltd, mf_dr)
kwargs['marker'] = markers[2]
kwargs['linestyle'] = linestyles[1]
ax.plot(x, data[:, 1], **kwargs)

kwargs = copy.deepcopy(line_kwargs)
kwargs['label'] = mf_dr + ltd
kwargs['label'] = '%s/%s' % (mf_dr + ltd, 'MF')
kwargs['marker'] = markers[3]
kwargs['linestyle'] = linestyles[2]
ax.plot(x, data[:, 2], **kwargs)

# kwargs = copy.deepcopy(line_kwargs)
# kwargs['label'] = nf_dr + ltd
# kwargs['marker'] = markers[4]
# kwargs['linestyle'] = linestyles[3]
# ax.plot(x, data[:, 3], **kwargs)

# ax.legend(loc='lower left', prop={'size': legend_size})
ax.legend(bbox_to_anchor=bbox_to_anchor,
          prop={'size': legend_size},
          mode='expand',
          loc=4,
          ncol=2)

ax.tick_params(axis='both', which='major', labelsize=tick_size)
ax.set_xlabel('Validation Set Size (\\%)', fontsize=label_size)
ax.set_ylabel('MSE', fontsize=label_size)

ax.set_xlim(x.min(), x.max())
ax.set_ylim(bottom=0.925)
ax.set_xticks(x)
xticklabels = ['0.2', '0.5', '1', '2', '5', '10', '20']
ax.set_xticklabels(xticklabels)

eps_file = path.join(fig_dir, run_file.replace('.py', '.eps'))
fig.savefig(eps_file, format='eps', bbox_inches='tight', pad_inches=pad_inches)