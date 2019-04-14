from os import path

import argparse
import pandas as pd

import read_data

def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('data_dir', type=str)
  parser.add_argument('result_file', type=str)
  return parser.parse_args()

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

  for u, test_item in test_data.items():
    rank_list = result_data[u]
    print(u)
    print(test_item)
    print(rank_list)
    input()
    # for u in self.test_users:
    #     user_ids = []
    #     user_neg_items = self.neg_items[u]
    #     item_ids = []
    #     # scores = []
    #     for j in user_neg_items:
    #         item_ids.append(j)
    #         user_ids.append(u)

    #     scores = self.predict(user_ids, item_ids)
    #     neg_item_index = list(zip(item_ids, scores))

    #     ranked_list[u] = sorted(neg_item_index, key=lambda tup: tup[1], reverse=True)
    #     pred_ratings[u] = [r[0] for r in ranked_list[u]]
    #     pred_ratings_5[u] = pred_ratings[u][:5]
    #     pred_ratings_10[u] = pred_ratings[u][:10]

    #     p_5, r_5, ndcg_5 = precision_recall_ndcg_at_k(5, pred_ratings_5[u], self.test_data[u])
    #     p_at_5.append(p_5)
    #     r_at_5.append(r_5)
    #     ndcg_at_5.append(ndcg_5)
    #     p_10, r_10, ndcg_10 = precision_recall_ndcg_at_k(10, pred_ratings_10[u], self.test_data[u])
    #     p_at_10.append(p_10)
    #     r_at_10.append(r_10)
    #     ndcg_at_10.append(ndcg_10)
    #     map_u, mrr_u, ndcg_u = map_mrr_ndcg(pred_ratings[u], self.test_data[u])
    #     map.append(map_u)
    #     mrr.append(mrr_u)
    #     ndcg.append(ndcg_u)
