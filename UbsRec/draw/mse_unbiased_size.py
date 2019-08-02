from ut_plot import *
from os import path

import copy
import matplotlib.pyplot as plt
import numpy as np

run_file = path.basename(__file__)
data_file = path.join(data_dir, run_file.replace('.py', '.dta'))
data = np.loadtxt(data_file, dtype=np.float32)

x = np.arange(data.shape[0])

fig, ax = plt.subplots(1, 1)
fig.set_size_inches(width, height, forward=True)

kwargs = copy.deepcopy(line_kwargs)
kwargs['label'] = mf_ips + ltd
kwargs['marker'] = markers[1]
kwargs['linestyle'] = name_linestyles['solid']
ax.plot(x, data[:, 0], **kwargs)

kwargs = copy.deepcopy(line_kwargs)
kwargs['label'] = nf_ips + ltd
kwargs['marker'] = markers[2]
kwargs['linestyle'] = name_linestyles['dotted']
ax.plot(x, data[:, 1], **kwargs)

kwargs = copy.deepcopy(line_kwargs)
kwargs['label'] = mf_dr + ltd
kwargs['marker'] = markers[3]
kwargs['linestyle'] = name_linestyles['densely dashed']
ax.plot(x, data[:, 2], **kwargs)

kwargs = copy.deepcopy(line_kwargs)
kwargs['label'] = nf_dr + ltd
kwargs['marker'] = markers[4]
kwargs['linestyle'] = name_linestyles['densely dashdotted']
ax.plot(x, data[:, 3], **kwargs)

ax.legend(bbox_to_anchor=bbox_to_anchor,
          prop={'size': legend_size},
          mode='expand',
          loc=4,
          ncol=2)

ax.tick_params(axis='both', which='major', labelsize=tick_size)
ax.set_xlabel('Validation Set Size (\\%)', fontsize=label_size)
ax.set_ylabel('MSE', fontsize=label_size)

ax.set_xlim(x.min(), x.max())
ax.set_xticks(x)
xticklabels = ['1', '5', '10', '20', '30', '40', '50']
ax.set_xticklabels(xticklabels)

eps_file = path.join(fig_dir, run_file.replace('.py', '.eps'))
fig.savefig(eps_file, format='eps', bbox_inches='tight', pad_inches=pad_inches)
