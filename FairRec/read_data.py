import pandas as pd
from os import path
from scipy.sparse import csr_matrix

def read_test(data_dir):
  test_file = path.join(data_dir, 'test.data')
  stat_file = path.join(data_dir, 'data.stat')
  names = ['u', 'i', 'r', 't']
  with open(stat_file, 'r') as fin:
    line = fin.readline()
    num_users = int(line.split('=')[-1])
    line = fin.readline()
    num_items = int(line.split('=')[-1])
  test_data = pd.read_csv(test_file, sep='\t', names=names)
  test_users = []
  test_items = []
  test_ratings = []
  for row in test_data.itertuples():
    test_users.append(row.u)
    test_items.append(row.i)
    test_ratings.append(1.0)
  test_data = csr_matrix((test_ratings, (test_users, test_items)),
                         shape=(num_users, num_items))
  test_dict = {}
  for u in range(num_users):
    test_dict[u] = test_data.getrow(u).nonzero()[1]
  test_data = test_dict
  return num_users, num_items, test_data