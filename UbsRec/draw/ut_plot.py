from os import path
from matplotlib import rc
from matplotlib import rcParams

import os

rc('font', **{'family': 'serif', 'serif': ['Times']})
rc('text', usetex=True)
rc('text.latex', unicode=True)
rc('legend', handlelength=rcParams['legend.handlelength'] * 0.7,
             handletextpad=rcParams['legend.handletextpad'] * 0.5)

data_dir = 'data'
fig_dir = 'figure'
if not path.exists(fig_dir):
  os.makedirs(fig_dir)

width = 6.4
height = 4.8
legend_size = 20
label_size = 20
line_width = 3.0
dotted_width = 3.5
marker_edge_width = 1.5
marker_size = 12
tick_size = 18
pad_inches = 0.10
bbox_to_anchor = (-0.03, 1.0, 1.06, 1.0)
c_kwargs = {'linewidth': line_width,
            'markersize': marker_size,
            'fillstyle': 'none',
            'markeredgewidth': marker_edge_width}
markers = ['x', 'v', 'o', 's']
colors = ['g', 'r', 'b', 'm']
linestyles = ['-', ':', '-.', '--']
mf_ips_np = 'MF-IPS-NP'
nfm_ips_np = 'NFM-IPS-NP'
mf_dr_np = 'MF-DR-NP'
nfm_dr_np = 'NFM-DR-NP'

def swap(l, i, j):
  t = l[i]
  l[i] = l[j]
  l[j] = t
  return l

