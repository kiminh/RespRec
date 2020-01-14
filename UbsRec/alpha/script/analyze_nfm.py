from os import path

import argparse

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('result_file', type=str)
  args = parser.parse_args()
  result_file = args.result_file

  with open(result_file, 'r') as fin:
    n_trial = 0
    line = True
    while True:
      if not line:
        break
      line = fin.readline()
      if line.startswith('Neural FM:'):
        param = line.strip()
        error = False
        while True:
          line = fin.readline()
          if not line:
            break
          if line.startswith('Best Iter(validation)='):
            error = line.strip()
            break
        if error:
          n_trial += 1
          fields = param.split(':')
          fields = fields[-1].split(',')
          fields = [f.strip() for f in fields]
          fields = [f for f in fields
                      if f.startswith('hidden_factor')
                      or f.startswith('dropout_keep')
                      or f.startswith('layers')
                      or f.startswith('lr')
                      or f[0] >= '0' and f[0] <= '1']
          param = ' '.join(fields)
          fields = error.split(' ')
          mae = fields[5]
          mse = fields[4]
          epoch = int(fields[2])
          print('%03d\t%s\t%s\t%s' % (epoch, mae, mse, param))
  print('#trial=%d' % (n_trial))


if __name__ == '__main__':
  main()



