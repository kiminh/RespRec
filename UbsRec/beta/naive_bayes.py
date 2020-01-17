from os import path

import argparse
import math
import numpy as np
import pandas as pd

valid_ratio = 0.05

def load_data_sets(data_dir):
  biased_file = path.join(data_dir, 'biased.dta')
  unbiased_file = path.join(data_dir, 'unbiased.dta')
  names = ['user', 'item', 'rating']
  biased_set = pd.read_csv(biased_file, sep='\t', names=names)
  unbiased_set = pd.read_csv(unbiased_file, sep='\t', names=names)

  train_set = biased_set
  n_unbiased = len(unbiased_set.index)
  n_valid = math.ceil(valid_ratio * n_unbiased)
  valid_set = unbiased_set[:n_valid]
  test_set = unbiased_set[n_valid:]
  return train_set, valid_set, test_set

def marginalize(data_set):
  n_rating = data_set.rating.unique().shape[0]
  marginal = np.zeros((n_rating))
  for rating in data_set.rating:
    marginal[rating - 1] += 1
  marginal = marginal / marginal.sum()
  for rating in range(n_rating):
    print('%d:%.4f' % (rating, marginal[rating]), end=' ')
  print('')
  return marginal

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('data_dir', type=str)
  args = parser.parse_args()

  data_dir = args.data_dir
  train_set, valid_set, test_set = load_data_sets(data_dir)

  train_marginal = marginalize(train_set)
  valid_marginal = marginalize(valid_set)
  test_marginal = marginalize(test_set)

  n_user = train_set.user.unique().shape[0]
  n_item = train_set.item.unique().shape[0]
  p_obs = len(train_set.index) / (n_user * n_item)
  propensities = p_obs * train_marginal / valid_marginal
  n_rating = train_set.rating.unique().shape[0]
  for rating in range(n_rating):
    print('%f' % (propensities[rating]), end='\t')
  print('')

if __name__ == '__main__':
  main()



