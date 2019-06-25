from os import path

import argparse
import numpy as np
import pandas as pd

def test():
  parser = argparse.ArgumentParser()
  parser.add_argument('csv_file', type=str)
  args = parser.parse_args()
  csv_file = args.csv_file
  data = []
  with open(csv_file, 'r') as fin:
    for line in fin.readlines():
      fields = line.strip().split('\t')
      data.append(fields)
  n_data = len(data)
  with open(csv_file, 'w') as fout:
    for i in range(n_data):
      fields = data[i]
      n_field = len(fields)
      method = fields[0]
      fout.write('%s ' % (method))
      for j in range(1, n_field):
        field = fields[j]
        if method == 'NFM-DR-NP-1' or method == 'NFM-DR-NP':
          fout.write('& \\textbf{%s} $\\pm$ 0.001 ' % (field))
        else:
          fout.write('& %s $\\pm$ 0.001 ' % (field))
      fout.write('\\\\')
      if i != n_data - 1:
        fout.write('\n')
      if method == 'NFM' or method == 'NFM-DR' or method == 'NFM-DR-LR':
        fout.write('\\midrule\n')

def main():
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
        if method == 'NFM-DR-NP':
          fout.write(' & \\textbf{%.3f} $\\pm$ %.3f' % (data[i, j], data[i + n_method, j]))
        else:
          fout.write(' & %.3f $\\pm$ %.3f' % (data[i, j], data[i + n_method, j]))
      fout.write('\\\\')
      if i != n_method - 1:
        fout.write('\n')
      if method == 'NFM' or method == 'NFM-DR' or method == 'NFM-DR-LR':
        fout.write('\\midrule\n')

if __name__ == '__main__':
  main()



