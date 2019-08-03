from os import path

import argparse
import numpy as np
import pandas as pd
import sys

def wt_variance():
  data_dir = ''
  run_file = path.basename(__file__)
  data_file = path.join(data_dir, run_file.replace('.py', '.dta'))
  temp_file = path.join(data_dir, run_file.replace('.py', '.tmp'))

  df = pd.read_csv(data_file, header=None, sep='\t', )
  data = df.values
  n_method = int(data.shape[0] / 2)
  n_value = int(data.shape[1] - 1)
  with open(temp_file, 'w') as fout:
    for i in range(n_method):
      method = data[i, 0]
      fout.write(method)
      for j in range(1, 1 + n_value):
        if method == 'NF-DR-LTD':
          fout.write(' & \\textbf{%.3f} $\\pm$ %.3f' % (data[i, j], data[i + n_method, j]))
        else:
          fout.write(' & %.3f $\\pm$ %.3f' % (data[i, j], data[i + n_method, j]))
      fout.write('\\\\')
      if i != n_method - 1:
        fout.write('\n')
      if method == 'NF' or method == 'NF-DR' or method == 'LR':
        fout.write('\\midrule\n')

def wo_variance():
  data_dir = ''
  run_file = path.basename(__file__)
  data_file = path.join(data_dir, run_file.replace('.py', '.dta'))
  temp_file = path.join(data_dir, run_file.replace('.py', '.tmp'))

  df = pd.read_csv(data_file, header=None, sep='\t', )
  data = df.values
  n_method = int(data.shape[0])
  n_value = int(data.shape[1] - 1)
  with open(temp_file, 'w') as fout:
    for i in range(n_method):
      method = data[i, 0]
      fout.write(method)
      for j in range(1, 1 + n_value):
        if method == 'NF-DR-LTD':
          fout.write(' & \\textbf{%.3f}' % (data[i, j]))
        else:
          fout.write(' & %.3f' % (data[i, j]))
      fout.write(' \\\\')
      if i != n_method - 1:
        fout.write('\n')
      if method == 'NF' or method == 'NF-DR':
        fout.write('\\midrule\n')

def propensity():
  data_dir = ''
  run_file = path.basename(__file__)
  data_file = path.join(data_dir, run_file.replace('.py', '.dta'))
  temp_file = path.join(data_dir, run_file.replace('.py', '.tmp'))

  df = pd.read_csv(data_file, header=None, sep='\t', )
  data = df.values
  n_method = int(data.shape[0])
  n_value = int(data.shape[1] - 1)
  with open(temp_file, 'w') as fout:
    for i in range(n_method):
      method = data[i, 0]
      if method == 'IP':
        fout.write('\\multirow{2}{*}{Existing} ')
      elif method == 'NP-0':
        fout.write('\\multirow{3}{*}{Proposed} ')
      fout.write('& %s' % (method))
      for j in range(1, 1 + n_value):
        if method == 'NP-1':
          fout.write(' & \\textbf{%.3f}' % (data[i, j]))
        else:
          fout.write(' & %.3f' % (data[i, j]))
      fout.write(' \\\\')
      if i != n_method - 1:
        fout.write('\n')
      if method == 'LR':
        fout.write('\\midrule\n')

if __name__ == '__main__':
  # wt_variance()
  # wo_variance()
  # propensity()
  parser = argparse.ArgumentParser()
  parser.add_argument('task', type=str)
  args = parser.parse_args()
  getattr(sys.modules[__name__], args.task)()


