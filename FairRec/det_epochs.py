from os import path

import matplotlib.pyplot as plt
import os

result_dir = 'result'
for file_name in os.listdir(result_dir):
  if len(file_name.split('_')) != 2:
    continue
  result_file = path.join(result_dir, file_name)
  x = []
  y = []
  with open(result_file, 'r') as fin:
    while True:
      line = fin.readline()
      if not line:
        break
      line = line.strip()
      if line.startswith('Epoch:'):
        start = line.index(' ') + 1
        stop = line.index(';')
        epoch = int(line[start:stop]) + 1
      if line.startswith('Final:'):
        start = line.index(' ') + 1
        stop = line.index(';')
        epoch = int(line[start:stop])
      if line.startswith('ndcg@10'):
        ndcg_at_10 = float(line.split(':')[1])
        x.append(epoch)
        y.append(ndcg_at_10)
  result_eps = result_file.replace('.tmp', '.eps')
  fig, ax = plt.subplots(1, 1)
  ax.plot(x, y)
  fig.savefig(result_eps, format='eps', bbox_inches='tight')
