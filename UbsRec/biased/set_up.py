'''Datasets that do not contain unbiased data
amazon: http://jmcauley.ucsd.edu/data/amazon/links.html
'''
from os import path

import argparse
import math
import numpy as np
import os
import pandas as pd

train_ratio = 0.80
valid_ratio = 0.05

def read_data_set(data_file, separator):
  data_set = []
  with open(data_file, 'r') as fin:
    while True:
      line = fin.readline()
      if not line:
        break
      fields = line.split(separator)
      user = fields[0]
      item = fields[1]
      rating = int(fields[2])
      data_set.append((user, item, rating))
  return data_set

def filter_data_set(data_file, separator, n_core):
  data_set = read_data_set(data_file, separator)
  n_rating = len(data_set)
  while True:
    user_n_rating = dict()
    for user, _, _ in data_set:
      user_n_rating[user] = user_n_rating.get(user, 0) + 1
    item_n_rating = dict()
    for _, item, _ in data_set:
      item_n_rating[item] = item_n_rating.get(item, 0) + 1
    invalid_users = set([user for user, n_rating in user_n_rating.items()
                              if n_rating < n_core])
    invalid_items = set([item for item, n_rating in item_n_rating.items()
                              if n_rating < n_core])
    if len(invalid_users) == 0 and len(invalid_items) == 0:
      break
    data_set = [(user, item, rating) for user, item, rating in data_set
                                     if user not in invalid_users
                                     and item not in invalid_items]
  return data_set

def save_data_set(data_set, data_dir):
  def _save_data_set(data_set, data_file):
    n_rating = len(data_set)
    with open(data_file, 'w') as fout:
      for user, item, rating in data_set:
        fout.write('%d\t%d\t%d\n' % (user, item, rating))
    print('Save %d ratings to %s' % (n_rating, data_file))

  if not path.exists(data_dir):
    os.makedirs(data_dir)
  n_rating = len(data_set)
  users = sorted(set([user for user, _, _ in data_set]))
  n_user = len(users)
  items = sorted(set([item for _, item, _ in data_set]))
  n_item = len(items)
  sparsity = 100 * n_rating / (n_user * n_item)
  print('#users %d . #items %d' % (n_user, n_item))
  print('#rating %d . sparsity %.2f%%' % (n_rating, sparsity))

  user_id = dict(zip(users, range(n_user)))
  item_id = dict(zip(items, range(n_item)))
  data_set = [(user_id[user], item_id[item], rating) for user, item, rating in data_set]

  n_train = math.ceil(train_ratio * n_rating)
  biased_set = data_set[:n_train]
  unbiased_set = data_set[n_train:]
  np.random.seed(0)
  np.random.shuffle(biased_set)
  np.random.shuffle(unbiased_set)
  biased_file = path.join(data_dir, 'biased.dta')
  unbiased_file = path.join(data_dir, 'unbiased.dta')
  _save_data_set(biased_set, biased_file)
  _save_data_set(unbiased_set, unbiased_file)

def shuffle_movie():
  def _maybe_download():
    if path.exists(raw_dir):
      return
    raw_url = 'http://files.grouplens.org/datasets/movielens/ml-1m.zip'
    par_dir = path.dirname(raw_dir)
    zip_file = path.join(par_dir, 'ml-1m.zip')
    os.system('wget %s -O %s' % (raw_url, zip_file))
    os.system('unzip %s -d %s' % (zip_file, par_dir))

  data_dir = 'movie'
  raw_dir = path.expanduser('~/Downloads/data/ml-1m')
  raw_file = path.join(raw_dir, 'ratings.dat')
  _maybe_download()
  data_set = read_data_set(raw_file, '::')
  save_data_set(data_set, data_dir)

def shuffle_book():
  def _maybe_download():
    if path.exists(raw_file):
      return
    os.system('wget %s -O %s' % (raw_url, raw_file))

  data_dir = 'book'
  amazon_url = 'http://snap.stanford.edu/data/amazon/productGraph/categoryFiles'
  raw_url = path.join(amazon_url, 'ratings_Books.csv')
  raw_file = path.expanduser('~/Downloads/data/book.csv')
  _maybe_download()
  data_set = filter_data_set(raw_file, ',', 10)
  save_data_set(data_set, data_dir)

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('data_set', choices=['book', 'movie'])
  parser.add_argument('out_format', choices=['lib', 'resp'])
  args = parser.parse_args()
  data_set = args.data_set
  out_format = args.out_format
  print('data_set %s . out_format %s' % (data_set, out_format))

  if data_set == 'book':
    shuffle_book()
  if data_set == 'movie':
    shuffle_movie()

if __name__ == '__main__':
  main()