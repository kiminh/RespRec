'''Datasets that do not contain unbiased data
amazon: http://jmcauley.ucsd.edu/data/amazon/links.html
'''
from os import path

import argparse
import math
import numpy as np
import os
import pandas as pd

train_ratio = 0.90
valid_ratio = 0.05
n_core = 10

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
      rating = fields[2]
      data_set.append((user, item, rating))
  return data_set

def filter_data_set(data_file, separator):
  data_set = read_data_set(data_file, separator)
  n_rating = len(data_set)
  print('Original #rating %d' % (n_rating))
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
    n_rating = len(data_set)
    users = set([user for user, _, _ in data_set])
    n_user = len(users)
    items = set([item for _, item, _ in data_set])
    n_item = len(items)
    print('Filtered #rating %d . #users %d . #items %d' % (n_rating, n_user, n_item))
  return data_set

def _filter_data_set(data_file, separator):
  data_set = read_data_set(data_file, separator)
  n_rating = len(data_set)
  print('Original #rating %d' % (n_rating))
  user_n_rating = dict()
  for user, _, _ in data_set:
    user_n_rating[user] = user_n_rating.get(user, 0) + 1
  invalid_users = set([user for user, n_rating in user_n_rating.items()
                            if n_rating < n_core])
  data_set = [(user, item, rating) for user, item, rating in data_set
                                   if user not in invalid_users]
  n_rating = len(data_set)
  users = set([user for user, _, _ in data_set])
  n_user = len(users)
  items = set([item for _, item, _ in data_set])
  n_item = len(items)
  print('Filtered #rating %d . #users %d . #items %d' % (n_rating, n_user, n_item))
  return data_set

def save_data_set(data_set, data_file):
  n_rating = len(data_set)
  with open(data_file, 'w') as fout:
    for user, item, rating in data_set:
      fout.write('%d\t%d\t%d\n' % (user, item, rating))
  print('Save %d ratings to %s' % (n_rating, data_file))

def shuffle_movie():
  def _maybe_download():
    if path.exists(raw_dir):
      return
    raw_url = 'http://files.grouplens.org/datasets/movielens/ml-1m.zip'
    par_dir = path.dirname(raw_dir)
    zip_file = path.join(par_dir, 'ml-1m.zip')
    os.system('wget %s -O %s' % (raw_url, zip_file))
    os.system('unzip %s -d %s' % (zip_file, par_dir))

  raw_dir = path.expanduser('~/Downloads/data/ml-1m')
  raw_file = path.join(raw_dir, 'ratings.dat')

  data_set = filter_data_set(raw_file, '::')

def shuffle_amazon(raw_url, raw_file):
  def _maybe_download():
    if path.exists(raw_file):
      return
    os.system('wget %s -O %s' % (raw_url, raw_file))

  _maybe_download()
  data_set = filter_data_set(raw_file, ',')

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('data_set', choices=['book', 'movie', 'tvshow'])
  parser.add_argument('out_format', choices=['lib', 'resp'])
  args = parser.parse_args()
  data_set = args.data_set
  out_format = args.out_format
  print('data_set %s . out_format %s' % (data_set, out_format))

  if data_set == 'movie':
    shuffle_movie()

  amazon_url = 'http://snap.stanford.edu/data/amazon/productGraph/categoryFiles'
  if data_set == 'book':
    raw_url = path.join(amazon_url, 'ratings_Books.csv')
    raw_file = path.expanduser('~/Downloads/data/book.csv')
    shuffle_amazon(raw_url, raw_file)
  if data_set == 'tvshow':
    raw_url = path.join(amazon_url, 'ratings_Movies_and_TV.csv')
    raw_file = path.expanduser('~/Downloads/data/tvshow.csv')
    shuffle_amazon(raw_url, raw_file)

if __name__ == '__main__':
  main()