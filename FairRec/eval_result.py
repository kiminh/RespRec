from os import path

import argparse
import math
import numpy as np
import pandas as pd

import read_data

def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('data_dir', type=str)
  parser.add_argument('result_file', type=str)
  return parser.parse_args()

def eval_prec(k, rank_list, test_item):
  test_item = set(test_item)
  count = 0.0
  for i in range(k):
    if rank_list[i] in test_item:
      count += 1.0
  return count / k

def eval_ndcg(k, rank_list, test_item):
  test_item = set(test_item)
  idcg = 0.0
  ik = min(k, len(test_item))
  for i in range(ik):
    idcg += 1.0 / math.log(i + 2, 2)
  dcg = 0.0
  for i in range(k):
    if rank_list[i] in test_item:
      dcg += 1.0 / math.log(i + 2, 2)
  return dcg / idcg

if __name__ == '__main__':
  args = parse_args()
  num_users, num_items, test_data = read_data.read_test(args.data_dir)
  user_file = path.join(args.data_dir, 'user.info')
  user_info = pd.read_csv(user_file, sep='\t', names=['u', 'g'])
  user_info = user_info.set_index('u').to_dict()['g']

  result_data = {}
  with open(args.result_file, 'r') as fin:
    while True:
      line = fin.readline()
      if not line:
        break
      fields = line.strip().split()
      u = int(fields[0])
      rank_list = [int(i) for i in fields[1:]]
      result_data[u] = rank_list

  prec_at_5_list = []
  prec_at_10_list = []
  ndcg_at_5_list = []
  ndcg_at_10_list = []
  for u, test_item in test_data.items():
    if u not in result_data:
      print(u)
      continue
    rank_list = result_data[u]
    prec_at_5 = eval_prec(5, rank_list, test_item)
    prec_at_5_list.append(prec_at_5)
    prec_at_10 = eval_prec(10, rank_list, test_item)
    prec_at_10_list.append(prec_at_10)
    ndcg_at_5 = eval_ndcg(5, rank_list, test_item)
    ndcg_at_5_list.append(ndcg_at_5)
    ndcg_at_10 = eval_ndcg(10, rank_list, test_item)
    ndcg_at_10_list.append(ndcg_at_10)
  prec_at_5 = np.mean(prec_at_5_list)
  prec_at_10 = np.mean(prec_at_10_list)
  ndcg_at_5 = np.mean(ndcg_at_5_list)
  ndcg_at_10 = np.mean(ndcg_at_10_list)
  print('precision@5: %.6f' % (prec_at_5))
  print('precision@10: %.6f' % (prec_at_10))
  print('ndcg@5: %.6f' % (ndcg_at_5))
  print('ndcg@10: %.6f' % (ndcg_at_10))




