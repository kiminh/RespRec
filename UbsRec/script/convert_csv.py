from os import path

import argparse

def main():
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
          fout.write('& \\textbf{%s} ' % (field))
        else:
          fout.write('& %s ' % (field))
      fout.write('\\\\')
      if i != n_data - 1:
        fout.write('\n')
      if method == 'NFM' or method == 'NFM-DR' or method == 'NFM-DR-LR':
        fout.write('\\midrule\n')


if __name__ == '__main__':
  main()


