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
legend_size = 25
label_size = 25
line_width = 3.5
dotted_width = 4.0
marker_edge_width = 1.5
marker_size = 12
tick_size = 23
pad_inches = 0.10
bbox_to_anchor = (-0.20, 1.0, 1.24, 1.0)
line_kwargs = {'linewidth': line_width,
               'markersize': marker_size,
               'fillstyle': 'none',
               'markeredgewidth': marker_edge_width}
bar_width = 0.25
capsize = 5
bar_kwargs = {'width': bar_width,
              'capsize': capsize}
markers = ['|', 'v', 's', 'p', 'o']
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

