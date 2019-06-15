from os import path

import argparse

def output(method, acc_list):
  mae = acc_list[0][0]
  mse = acc_list[0][1]
  pct = 100 * mae / mse
  params = acc_list[0][2]
  p_data = (method, mae, mse, pct, params)
  print(' %-8s & %.3f & %.3f & %.2f%% & %s' % p_data)

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('result_file', type=str)
  args = parser.parse_args()

  result_file = args.result_file
  method_acc = {}
  with open(result_file, 'r') as fin:
    while True:
      line = fin.readline()
      if not line:
        break
      fields = line.split()
      method = fields[0].split('_')[0]
      if method not in method_acc:
        method_acc[method] = []
      mae = float(fields[1])
      mse = float(fields[2])
      method_acc[method].append((mae, mse, fields[0]))
  methods = sorted(method_acc.keys())
  print(path.basename(result_file))
  for method in methods:
    acc_list = method_acc[method]
    acc_list = sorted(acc_list, key=lambda e: (e[0], e[1]))
    output(method, acc_list)
    acc_list = sorted(acc_list, key=lambda e: (e[1], e[0]))
    output(method, acc_list)

if __name__ == '__main__':
  main()



