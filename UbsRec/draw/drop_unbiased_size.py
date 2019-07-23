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

x = np.arange(data.shape[0])

fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
fig.set_size_inches(width, height, forward=True)

kwargs = copy.deepcopy(line_kwargs)
kwargs['label'] = '%s/%s' % (nf_dr , nf)
kwargs['marker'] = markers[1]
ax1.plot(x, data[:, 0], **kwargs)
ax2.plot(x, data[:, 0], **kwargs)

kwargs = copy.deepcopy(line_kwargs)
kwargs['label'] = '%s/%s' % (nf_dr + ltd, nf_dr)
kwargs['marker'] = markers[2]
ax1.plot(x, data[:, 2], **kwargs)
ax2.plot(x, data[:, 2], **kwargs)

kwargs = copy.deepcopy(line_kwargs)
kwargs['label'] = '%s/%s' % (nf_dr + ltd, nf)
kwargs['marker'] = markers[3]
ax1.plot(x, data[:, 1], **kwargs)
ax2.plot(x, data[:, 1], **kwargs)

margin = 0.006
ax1.set_ylim(data[:, 0:2].min() - margin, data[:, 0:2].max() + margin)
ax2.set_ylim(data[:, 2].min() - margin, data[:, 2].max() + margin)
ax1.spines['bottom'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax1.xaxis.tick_top()
ax1.tick_params(labeltop='off')
ax2.xaxis.tick_bottom()
kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False)
ax1.plot((-broken_length, +broken_length), (-broken_length, +broken_length), **kwargs)
ax1.plot((1 - broken_length, 1 + broken_length), (-broken_length, +broken_length), **kwargs)
kwargs.update(transform=ax2.transAxes)
ax2.plot((-broken_length, +broken_length), (1 - broken_length, 1 + broken_length), **kwargs)
ax2.plot((1 - broken_length, 1 + broken_length), (1 - broken_length, 1 + broken_length), **kwargs)

ax1.legend(bbox_to_anchor=bbox_to_anchor,
          prop={'size': legend_size},
          mode='expand',
          loc=4,
          ncol=2)

ax1.set_xlim(x.min(), x.max())
ax2.set_xlim(x.min(), x.max())
ax1.tick_params(axis='both', which='major', labelsize=tick_size)
ax2.tick_params(axis='both', which='major', labelsize=tick_size)
ax1.tick_params(axis='x', which='both', bottom=False, top=False)
fig.text(0.0, 0.7, 'Drop in MSE (\\%)', rotation='vertical', fontsize=label_size)
ax2.set_xlabel('Unbiased Set Size (\\%)', fontsize=label_size)
ax2.set_xticks(x)
ax2.set_xticklabels(['1', '5', '10', '20', '30', '40', '50'])
ax1_yticks = [0.35, 0.40]
ax1.set_yticks(ax1_yticks)
ax1.set_yticklabels(['%d' % (ax1_ytick * 100) for ax1_ytick in ax1_yticks])
ax2_yticks = [0.04, 0.05]
ax2.set_yticks(ax2_yticks)
ax2.set_yticklabels(['%d' % (ax2_ytick * 100) for ax2_ytick in ax2_yticks])


eps_file = path.join(fig_dir, run_file.replace('.py', '.eps'))
fig.savefig(eps_file, format='eps', bbox_inches='tight', pad_inches=pad_inches)


