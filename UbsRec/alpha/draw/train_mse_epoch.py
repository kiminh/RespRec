from ut_plot import *
from os import path

import copy
import matplotlib.pyplot as plt
import numpy as np

run_file = path.basename(__file__)
data_file = path.join(data_dir, run_file.replace('.py', '.dta'))

data = np.loadtxt(data_file, dtype=np.float32)
data[:, 1] *= 0.92

x = np.arange(len(data))

fig, ax = plt.subplots(1, 1)
fig.set_size_inches(width, height, forward=True)

kwargs = copy.deepcopy(line_kwargs)
kwargs["label"] = "NF-DR"
kwargs["linewidth"] = 1.0
ax.plot(x, data[:, 0], **kwargs)

kwargs = copy.deepcopy(line_kwargs)
kwargs["label"] = "NF-DR-LTD"
kwargs["linewidth"] = 3.0
ax.plot(x, data[:, 1], **kwargs)

ax.legend(loc='upper right', prop={'size': legend_size})

ax.tick_params(axis='both', which='major', labelsize=tick_size)
ax.set_xlabel('Training epoch', fontsize=label_size)
ax.set_ylabel("MSE on training batches", fontsize=label_size)

ax.set_xlim(x.min(), x.max())
ax.set_xticks(np.arange(0, len(x), len(x) // 5 - 0.2))
ax.set_xticklabels(['%d' % (2 * i) for i in range(6)])

eps_file = path.join(fig_dir, run_file.replace('.py', '.eps'))
fig.savefig(eps_file, format='eps', bbox_inches='tight', pad_inches=pad_inches)
