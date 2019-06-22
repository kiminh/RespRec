from ut_plot import *
from os import path

import copy
import matplotlib.pyplot as plt
import numpy as np

run_file = path.basename(__file__)
data_file = path.join(data_dir, run_file.replace('.py', '.dta'))
data = np.loadtxt(data_file, dtype=np.float32)

n_rating = data.shape[1]

f = (data[1, :].sum() - data[0, :].sum()) / (n_rating * data[1, 4])
data[1, :] -= data[1, 4] * f
data[2, :] *= (1.0 - f)
f = (data[3, :].sum() - data[0, :].sum()) / (n_rating * data[3, 4])
data[3, :] -= data[3, 4] * f
data[4, :] *= (1.0 - f)

print(data[1, :])
print(data[3, :])

fig, ax = plt.subplots(1, 1)
fig.set_size_inches(width, height, forward=True)

width = 0.25
capsize = 5
x = np.arange(1, 1 + n_rating)
kwargs = copy.deepcopy(bar_kwargs)
kwargs['label'] = 'NB'
plt.bar(x - width, data[0, :], **kwargs)
kwargs = copy.deepcopy(bar_kwargs)
kwargs['yerr'] = data[2, :]
kwargs['label'] = '$\\lambda=0$'
plt.bar(x, data[1, :], **kwargs)
kwargs = copy.deepcopy(bar_kwargs)
kwargs['yerr'] = data[4, :]
kwargs['label'] = '$\\lambda=1$'
plt.bar(x + width, data[3, :], **kwargs)

ax.legend(bbox_to_anchor=bbox_to_anchor,
          prop={'size': legend_size},
          mode='expand',
          loc=4,
          ncol=3)

ax.tick_params(axis='both', which='major', labelsize=tick_size)
ax.set_xlabel('Rating Value', fontsize=label_size)
ax.set_ylabel('Propensity', fontsize=label_size)
ax.set_xticks(np.arange(1, 5.5, 1))
ax.set_yticks(np.arange(0.00, 0.175, 0.05))

eps_file = path.join(fig_dir, run_file.replace('.py', '.eps'))
fig.savefig(eps_file, format='eps', bbox_inches='tight', pad_inches=pad_inches)
