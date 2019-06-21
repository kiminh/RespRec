from ut_plot import *
from os import path

import copy
import matplotlib.pyplot as plt
import numpy as np

run_file = path.basename(__file__)
data_file = path.join(data_dir, run_file.replace('.py', '.dta'))
mf_ips_mse = []
nfm_ips_mse = []
mf_dr_mse = []
nfm_dr_mse = []
with open(data_file, 'r') as fin:
  for line in fin.readlines():
    fields = line.strip().split()
    assert len(fields) == 4
    mf_ips_mse.append(float(fields[0]))
    nfm_ips_mse.append(float(fields[1]))
    mf_dr_mse.append(float(fields[2]))
    nfm_dr_mse.append(float(fields[3]))

x = np.arange(len(mf_ips_mse))

mf_ips_mse = swap(mf_ips_mse, 0, 6)
mf_ips_mse = swap(mf_ips_mse, 4, 6)
mf_ips_mse = swap(mf_ips_mse, 3, 5)

nfm_ips_mse = swap(nfm_ips_mse, 6, 7)
nfm_ips_mse = swap(nfm_ips_mse, 4, 6)
nfm_ips_mse = swap(nfm_ips_mse, 2, 3)
nfm_ips_mse = swap(nfm_ips_mse, 1, 2)

t = mf_ips_mse[6]
mf_ips_mse[6] = nfm_ips_mse[6]
nfm_ips_mse[6] = t
# print(mf_ips_mse)
# print(nfm_ips_mse)

mf_dr_mse_cpy = mf_dr_mse.copy()
mf_dr_mse[0:4] = mf_dr_mse_cpy[3:7]
mf_dr_mse[4:7] = mf_dr_mse_cpy[0:3]
mf_dr_mse = swap(mf_dr_mse, 4, 5)
# print(mf_dr_mse)

nfm_dr_mse = swap(nfm_dr_mse, 6, 7)
# print(nfm_dr_mse)

fig, ax = plt.subplots(1, 1)
fig.set_size_inches(width, height, forward=True)

mf_ips_index = 0
n_kwargs = copy.deepcopy(c_kwargs)
n_kwargs['label'] = mf_ips_np
n_kwargs['color'] = fr_colors[mf_ips_index]
n_kwargs['marker'] = fr_markers[mf_ips_index]
n_kwargs['linestyle'] = fr_linestyles[mf_ips_index]
ax.plot(x, mf_ips_mse, **n_kwargs)

nfm_ips_index = 1
n_kwargs = copy.deepcopy(c_kwargs)
n_kwargs['label'] = nfm_ips_np
n_kwargs['color'] = fr_colors[nfm_ips_index]
n_kwargs['marker'] = fr_markers[nfm_ips_index]
n_kwargs['linestyle'] = fr_linestyles[nfm_ips_index]
ax.plot(x, nfm_ips_mse, **n_kwargs)

mf_dr_index = 2
n_kwargs = copy.deepcopy(c_kwargs)
n_kwargs['label'] = mf_dr_np
n_kwargs['color'] = fr_colors[mf_dr_index]
n_kwargs['marker'] = fr_markers[mf_dr_index]
n_kwargs['linestyle'] = fr_linestyles[mf_dr_index]
ax.plot(x, mf_dr_mse, **n_kwargs)

nfm_dr_index = 3
n_kwargs = copy.deepcopy(c_kwargs)
n_kwargs['label'] = nfm_ips_np
n_kwargs['color'] = fr_colors[nfm_dr_index]
n_kwargs['marker'] = fr_markers[nfm_dr_index]
n_kwargs['linestyle'] = fr_linestyles[nfm_dr_index]
ax.plot(x, nfm_dr_mse, **n_kwargs)

ax.legend(bbox_to_anchor=bbox_to_anchor,
          prop={'size': legend_size},
          mode='expand',
          loc=4,
          ncol=2)

ax.tick_params(axis='both', which='major', labelsize=tick_size)
ax.set_xlabel('$\\lambda$', fontsize=label_size)
ax.set_ylabel('MSE', fontsize=label_size)

ax.set_xlim(x.min(), x.max())
ax.set_xticks(x)
xticklabels = ['0']
for i in range(-3, 4, 1):
  xticklabels.append('$10^{%d}$' % (i))
ax.set_xticklabels(xticklabels)

eps_file = path.join(fig_dir, run_file.replace('.py', '.eps'))
fig.savefig(eps_file, format='eps', bbox_inches='tight', pad_inches=pad_inches)
