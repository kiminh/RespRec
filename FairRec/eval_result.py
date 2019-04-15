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
  assert len(test_item) > 0
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
  user_file = path.join(args.data_dir, 'user.attr')
  user_attr = read_data.read_attr(user_file)

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

  attr_set = set()
  prec_at_5_list = []
  prec_at_10_list = []
  ndcg_at_5_list = []
  ndcg_at_10_list = []
  prec_at_5_dict = {}
  prec_at_10_dict = {}
  ndcg_at_5_dict = {}
  ndcg_at_10_dict = {}
  for u, test_item in test_data.items():
    if len(test_item) == 0:
      continue
    if u not in result_data:
      print('user %d does not exist on result data' % (u))
      continue
    rank_list = result_data[u]
    prec_at_5 = eval_prec(5, rank_list, test_item)
    prec_at_10 = eval_prec(10, rank_list, test_item)
    ndcg_at_5 = eval_ndcg(5, rank_list, test_item)
    ndcg_at_10 = eval_ndcg(10, rank_list, test_item)
    attr = user_attr[u]
    attr_set.add(attr)
    prec_at_5_list.append(prec_at_5)
    prec_at_10_list.append(prec_at_10)
    ndcg_at_5_list.append(ndcg_at_5)
    ndcg_at_10_list.append(ndcg_at_10)
    if attr not in prec_at_5_dict:
      prec_at_5_dict[attr] = []
      prec_at_10_dict[attr] = []
      ndcg_at_5_dict[attr] = []
      ndcg_at_10_dict[attr] = []
    prec_at_5_dict[attr].append(prec_at_5)
    prec_at_10_dict[attr].append(prec_at_10)
    ndcg_at_5_dict[attr].append(ndcg_at_5)
    ndcg_at_10_dict[attr].append(ndcg_at_10)
  prec_at_5 = np.mean(prec_at_5_list)
  prec_at_10 = np.mean(prec_at_10_list)
  ndcg_at_5 = np.mean(ndcg_at_5_list)
  ndcg_at_10 = np.mean(ndcg_at_10_list)
  # print(' prec@5: %.6f' % (prec_at_5))
  # print('prec@10: %.6f' % (prec_at_10))
  # print(' ndcg@5: %.6f' % (ndcg_at_5))
  # print('ndcg@10: %.6f' % (ndcg_at_10))

  min_attr = None
  min_prec_at_5 = 1.0
  min_prec_at_10 = 1.0
  min_ndcg_at_5 = 1.0
  min_ndcg_at_10 = 1.0
  for attr in sorted(attr_set):
    prec_at_10 = np.mean(prec_at_10_dict[attr])
    if prec_at_10 < min_prec_at_10:
      min_attr = attr
      min_prec_at_5 = np.mean(prec_at_5_dict[attr])
      min_prec_at_10 = np.mean(prec_at_10_dict[attr])
      min_ndcg_at_5 = np.mean(ndcg_at_5_dict[attr])
      min_ndcg_at_10 = np.mean(ndcg_at_10_dict[attr])

  print(path.basename(args.result_file))
  for attr in sorted(attr_set):
    print('%s' % (attr))
    if attr == min_attr:
      print('\t', ' prec@5: %.6f' % (np.mean(prec_at_5_dict[attr])))
      print('\t', 'prec@10: %.6f' % (np.mean(prec_at_10_dict[attr])))
      print('\t', ' ndcg@5: %.6f' % (np.mean(ndcg_at_5_dict[attr])))
      print('\t', 'ndcg@10: %.6f' % (np.mean(ndcg_at_10_dict[attr])))
    else:
      prec_at_5 = np.mean(prec_at_5_dict[attr])
      prec_at_5_pct = (prec_at_5 - min_prec_at_5) / min_prec_at_5 * 100
      print('\t', ' prec@5: %.6f (%.2f%%)' % (prec_at_5, prec_at_5_pct))
      prec_at_10 = np.mean(prec_at_10_dict[attr])
      prec_at_10_pct = (prec_at_10 - min_prec_at_10) / min_prec_at_10 * 100
      print('\t', 'prec@10: %.6f (%.2f%%)' % (prec_at_10, prec_at_10_pct))
      ndcg_at_5 = np.mean(ndcg_at_5_dict[attr])
      ndcg_at_5_pct = (ndcg_at_5 - min_ndcg_at_5) / min_ndcg_at_5 * 100
      print('\t', ' ndcg@5: %.6f (%.2f%%)' % (ndcg_at_5, ndcg_at_5_pct))
      ndcg_at_10 = np.mean(ndcg_at_10_dict[attr])
      ndcg_at_10_pct = (ndcg_at_10 - min_ndcg_at_10) / min_ndcg_at_10 * 100
      print('\t', 'ndcg@10: %.6f (%.2f%%)' % (ndcg_at_10, ndcg_at_10_pct))
  print('')
