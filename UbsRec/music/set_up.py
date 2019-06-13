from os import path

import argparse
import numpy as np
import os
import pandas as pd

def load_data_set(data_file):
  read_kwargs = {'sep': '\t', 'names': ['user', 'item', 'rating']}
  rating_df = pd.read_csv(data_file, **read_kwargs)
  rating_df.user = rating_df.user - 1
  rating_df.item = rating_df.item - 1
  n_user = rating_df.user.unique().shape[0]
  n_item = rating_df.item.unique().shape[0]
  return n_user, n_item, rating_df

def save_data_set(data_set, data_file):
  to_kwargs = {'sep': '\t', 'header': False, 'index': False, 
               'columns': ['user', 'item', 'rating']}
  data_set.to_csv(data_file, **to_kwargs)

def shuffle_data(in_dir):
  data_dir = 'data'
  if path.exists(data_dir):
    return
  biased_file = path.join(in_dir, 'ydata-ymusic-rating-study-v1_0-train.txt')
  unbiased_file = path.join(in_dir, 'ydata-ymusic-rating-study-v1_0-test.txt')
  n_user, n_item, biased_set = load_data_set(biased_file)
  _, _, unbiased_set = load_data_set(unbiased_file)
  print('#user=%d #item=%d' % (n_user, n_item))
  n_biased = biased_set.shape[0]
  n_unbiased = unbiased_set.shape[0]
  print('#biased=%d #unbiased=%d' % (n_biased, n_unbiased))

  min_user = biased_set.user.min()
  max_user = biased_set.user.max()
  print('user=[%d, %d]' % (min_user, max_user))
  min_item = biased_set.item.min()
  max_item = biased_set.item.max()
  print('item=[%d, %d]' % (min_item, max_item))

  biased_set = biased_set.sample(frac=1, random_state=0)
  unbiased_set = unbiased_set.sample(frac=1, random_state=0)
  os.makedirs(data_dir)
  biased_file = path.join(data_dir, 'biased.dta')
  unbiased_file = path.join(data_dir, 'unbiased.dta')
  save_data_set(biased_set, biased_file)
  save_data_set(unbiased_set, unbiased_file)

def to_lib_once(ubs_ratio, inc_valid):
  pass

def to_lib_many(ubs_ratio):
  to_lib_once(ubs_ratio, False)
  to_lib_once(ubs_ratio, True)

def to_coat_once(ubs_ratio, inc_valid):
  pass

def to_resp_many(ubs_ratio):
  to_coat_once(ubs_ratio, False)
  to_coat_once(ubs_ratio, True)

def main():
  in_dir = path.expanduser('~/Downloads/data/Webscope_R3')
  if not path.exists(in_dir):
    raise Exception('Please download the music dataset from Yahoo.')
  shuffle_data(in_dir)

  parser = argparse.ArgumentParser()
  parser.add_argument('out_format', choices=['lib', 'resp'])
  parser.add_argument('ubs_ratio', type=float)
  args = parser.parse_args()
  out_format = args.out_format
  ubs_ratio = args.ubs_ratio
  if out_format == 'lib':
    to_lib_many(ubs_ratio)
  if out_format == 'resp':
    to_resp_many(ubs_ratio)

if __name__ == '__main__':
  main()