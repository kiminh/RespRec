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
book_core = 25 # 36
movie_core = 216

def read_data_set(data_file, separator):
  data_set = []
  rating_set = set()
  with open(data_file, 'r') as fin:
    while True:
      line = fin.readline()
      if not line:
        break
      fields = line.split(separator)
      user = fields[0]
      item = fields[1]
      rating = int(float(fields[2]))
      rating_set.add(fields[2])
      data_set.append((user, item, rating))
  print(rating_set)
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
    users = sorted(set([user for user, _, _ in data_set]))
    n_user = len(users)
    items = sorted(set([item for _, item, _ in data_set]))
    n_item = len(items)
    print('#Users %d . #Items %d' % (n_user, n_item))
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
  print('#Users %d . #Items %d' % (n_user, n_item))
  print('#Ratings %d . Sparsity %.2f%%' % (n_rating, sparsity))

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

def shuffle_book():
  def _maybe_download():
    if path.exists(raw_file):
      return
    os.system('wget %s -O %s' % (raw_url, raw_file))

  data_dir = 'book'
  # if path.exists(data_dir):
  #   return
  amazon_url = 'http://snap.stanford.edu/data/amazon/productGraph/categoryFiles'
  raw_url = path.join(amazon_url, 'ratings_Books.csv')
  raw_file = path.expanduser('~/Downloads/data/book.csv')
  _maybe_download()
  data_set = filter_data_set(raw_file, ',', book_core)
  save_data_set(data_set, data_dir)

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
  # if path.exists(data_dir):
  #   return
  raw_dir = path.expanduser('~/Downloads/data/ml-1m')
  raw_file = path.join(raw_dir, 'ratings.dat')
  _maybe_download()
  data_set = filter_data_set(raw_file, '::', movie_core)
  save_data_set(data_set, data_dir)

def stringify(number):
  string = '%f' % (number)
  string = string.rstrip('0')
  string = string[:-1] if string.endswith('.') else string
  return string

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

def to_lib_once(data_name, inc_valid):
  def _to_lib_once(ratings, out_dir):
    kwargs = {'sep': '\t', 'header': False, 'index':False}
    if not path.exists(out_dir):
      os.makedirs(out_dir)
    data_file = path.join(out_dir, 'ratings.txt')
    n_rating = len(ratings.index)
    print('Save %d ratings to %s' % (n_rating, data_file))
    ratings.to_csv(data_file, **kwargs)

  base_dir = path.expanduser('~/Projects/librec/data')
  dir_name = data_name
  dir_name += '_incl' if inc_valid else '_excl'
  dir_name += '_%s' % (stringify(valid_ratio))
  out_dir = path.join(base_dir, dir_name)
  if not path.exists(out_dir):
    os.makedirs(out_dir)

  train_set, valid_set, test_set = load_data_sets(data_name)
  if inc_valid:
    train_set = pd.concat([train_set, valid_set])

  train_dir = path.join(out_dir, 'train')
  test_dir = path.join(out_dir, 'test')
  _to_lib_once(train_set, train_dir)
  _to_lib_once(test_set, test_dir)

def to_lib_many(data_name):
  to_lib_once(data_name, False)
  to_lib_once(data_name, True)

def to_resp_once(data_name, inc_valid):
  def _to_resp_once(data_set, data_file):
    with open(data_file, 'w') as fout:
      for row in data_set.itertuples():
        fout.write('%d' % (row.rating))
        fout.write(' %d:1' % (user_ids[row.user]))
        fout.write(' %d:1\n' % (item_ids[row.item]))
    n_rating = len(data_set.index)
    print('Save %d ratings to %s' % (n_rating, data_file))

  base_dir = path.expanduser('~/Downloads/data')
  dir_name = data_name
  dir_name += '_incl' if inc_valid else '_excl'
  dir_name += '_%s' % (stringify(valid_ratio))
  out_dir = path.join(base_dir, dir_name)
  if not path.exists(out_dir):
    os.makedirs(out_dir)

  train_set, valid_set, test_set = load_data_sets(data_name)
  if inc_valid:
    train_set = pd.concat([train_set, valid_set])

  global_id = 0
  users = set(train_set.user.unique())
  users = users.union(set(valid_set.user.unique()))
  users = users.union(set(test_set.user.unique()))
  user_ids = dict()
  for user in sorted(users):
    user_ids[user] = global_id
    global_id += 1
  items = set(train_set.item.unique())
  items = items.union(set(valid_set.item.unique()))
  items = items.union(set(test_set.item.unique()))
  item_ids = dict()
  for item in sorted(items):
    item_ids[item] = global_id
    global_id += 1

  train_file = path.join(out_dir, '%s.train.libfm' % (dir_name))
  valid_file = path.join(out_dir, '%s.validation.libfm' % (dir_name))
  test_file = path.join(out_dir, '%s.test.libfm' % (dir_name))
  _to_resp_once(train_set, train_file)
  _to_resp_once(valid_set, valid_file)
  _to_resp_once(test_set, test_file)

def to_resp_many(data_name):
  to_resp_once(data_name, False)
  to_resp_once(data_name, True)

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('data_name', choices=['book', 'movie'])
  parser.add_argument('out_format', choices=['lib', 'resp'])
  args = parser.parse_args()
  data_name = args.data_name
  out_format = args.out_format

  if data_name == 'book':
    shuffle_book()
  if data_name == 'movie':
    shuffle_movie()
  if out_format == 'lib':
    to_lib_many(data_name)
  if out_format == 'resp':
    to_resp_many(data_name)

if __name__ == '__main__':
  main()