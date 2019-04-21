#!/usr/bin/env python
"""
Evaluation Metrics for Top N Recommendation
"""

import numpy as np

__author__ = "Shuai Zhang"
__copyright__ = "Copyright 2018, The DeepRec Project"

__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Shuai Zhang"
__email__ = "cheungdaven@gmail.com"
__status__ = "Development"

import math


# efficient version
def precision_recall_ndcg_at_k(k, rankedlist, test_matrix):
    idcg_k = 0
    dcg_k = 0
    n_k = k if len(test_matrix) > k else len(test_matrix)
    for i in range(n_k):
        idcg_k += 1 / math.log(i + 2, 2)

    b1 = rankedlist
    b2 = test_matrix
    s2 = set(b2)
    hits = [(idx, val) for idx, val in enumerate(b1) if val in s2]
    count = len(hits)

    for c in range(count):
        dcg_k += 1 / math.log(hits[c][0] + 2, 2)

    return float(count / k), float(count / len(test_matrix)), float(dcg_k / idcg_k)


def map_mrr_ndcg(rankedlist, test_matrix):
    ap = 0
    map = 0
    dcg = 0
    idcg = 0
    mrr = 0
    for i in range(len(test_matrix)):
        idcg += 1 / math.log(i + 2, 2)

    b1 = rankedlist
    b2 = test_matrix
    s2 = set(b2)
    hits = [(idx, val) for idx, val in enumerate(b1) if val in s2]
    count = len(hits)

    for c in range(count):
        ap += (c + 1) / (hits[c][0] + 1)
        dcg += 1 / math.log(hits[c][0] + 2, 2)

    if count != 0:
        mrr = 1 / (hits[0][0] + 1)

    if count != 0:
        map = ap / count

    return map, mrr, float(dcg / idcg)

def average(metric):
  metric = {k: np.mean(v) for k, v in metric.items()}
  return metric

def evaluate(self):
  pred_ratings_10 = {}
  pred_ratings_5 = {}
  pred_ratings = {}
  ranked_list = {}
  map = {}
  mrr = {}
  ndcg = {}
  p_at_5 = {}
  r_at_5 = {}
  ndcg_at_5 = {}
  p_at_10 = {}
  r_at_10 = {}
  ndcg_at_10 = {}
  for u in self.test_users:
    a = self.user_attr[u]
    user_ids = []
    user_neg_items = self.neg_items[u]
    item_ids = []
    # scores = []
    for j in user_neg_items:
      item_ids.append(j)
      user_ids.append(u)

    scores = self.predict(user_ids, item_ids)
    # print(type(scores))
    # print(scores)
    # print(np.shape(scores))
    # print(ratings)
    neg_item_index = list(zip(item_ids, scores))

    ranked_list[u] = sorted(neg_item_index, key=lambda tup: tup[1], reverse=True)
    pred_ratings[u] = [r[0] for r in ranked_list[u]]
    pred_ratings_5[u] = pred_ratings[u][:5]
    pred_ratings_10[u] = pred_ratings[u][:10]

    p_5, r_5, ndcg_5 = precision_recall_ndcg_at_k(5, pred_ratings_5[u], self.test_data[u])
    p_10, r_10, ndcg_10 = precision_recall_ndcg_at_k(10, pred_ratings_10[u], self.test_data[u])
    map_u, mrr_u, ndcg_u = map_mrr_ndcg(pred_ratings[u], self.test_data[u])
    if a not in map:
      map[a] = []
      mrr[a] = []
      ndcg[a] = []
      p_at_5[a] = []
      r_at_5[a] = []
      ndcg_at_5[a] = []
      p_at_10[a] = []
      r_at_10[a] = []
      ndcg_at_10[a] = []
    map[a].append(map_u)
    mrr[a].append(mrr_u)
    ndcg[a].append(ndcg_u)
    p_at_5[a].append(p_5)
    r_at_5[a].append(r_5)
    ndcg_at_5[a].append(ndcg_5)
    p_at_10[a].append(p_10)
    r_at_10[a].append(r_10)
    ndcg_at_10[a].append(ndcg_10)

  a_set = set(self.user_attr.values())
  a_list = sorted(a_set)
  print('')
  print('-' * 32)
  for a in a_list:
    print('    map: %.6f (%s)' % (np.mean(map[a]), a))
    print('    mrr: %.6f (%s)' % (np.mean(mrr[a]), a))
    print('   ndcg: %.6f (%s)' % (np.mean(ndcg[a]), a))
    print(' prec@5: %.6f (%s)' % (np.mean(p_at_5[a]), a))
    print('  rec@5: %.6f (%s)' % (np.mean(r_at_5[a]), a))
    print(' ndcg@5: %.6f (%s)' % (np.mean(ndcg_at_5[a]), a))
    print('prec@10: %.6f (%s)' % (np.mean(p_at_10[a]), a))
    print(' rec@10: %.6f (%s)' % (np.mean(r_at_10[a]), a))
    print('ndcg@10: %.6f (%s)' % (np.mean(ndcg_at_10[a]), a))
  print('-' * 32)
  print('')
  map = average(map)
  mrr = average(mrr)
  ndcg = average(ndcg)
  p_at_5 = average(p_at_5)
  r_at_5 = average(r_at_5)
  ndcg_at_5 = average(ndcg_at_5)
  p_at_10 = average(p_at_10)
  r_at_10 = average(r_at_10)
  ndcg_at_10 = average(ndcg_at_10)
  return map, mrr, ndcg, p_at_5, r_at_5, ndcg_at_5, p_at_10, r_at_10, ndcg_at_10



