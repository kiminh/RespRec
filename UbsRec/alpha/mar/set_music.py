from os import path

import argparse
import math
import numpy as np
import os
import pandas as pd

dflt_ratio = 0.05
data_dir = 'music'
# max_ratio = 0.2
max_ratio = 0.5

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
  if not path.exists(data_dir):
    os.makedirs(data_dir)
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
  biased_file = path.join(data_dir, 'biased.dta')
  unbiased_file = path.join(data_dir, 'unbiased.dta')
  save_data_set(biased_set, biased_file)
  save_data_set(unbiased_set, unbiased_file)

def stringify(number):
  string = '%f' % (number)
  string = string.rstrip('0')
  string = string[:-1] if string.endswith('.') else string
  return string

def load_data_sets(valid_ratio):
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

def to_lib_once(inc_valid):
  def _to_lib_once(ratings, out_dir):
    kwargs = {'sep': '\t', 'header': False, 'index':False}
    if not path.exists(out_dir):
      os.makedirs(out_dir)
    data_file = path.join(out_dir, 'ratings.txt')
    n_rating = len(ratings.index)
    print('Save %d ratings to %s' % (n_rating, data_file))
    ratings.to_csv(data_file, **kwargs)

  base_dir = path.expanduser('~/Projects/librec/data')
  dir_name = 'music'
  dir_name += '_incl' if inc_valid else '_excl'
  dir_name += '_%s' % (stringify(dflt_ratio))
  out_dir = path.join(base_dir, dir_name)

  train_set, valid_set, test_set = load_data_sets(dflt_ratio)
  if inc_valid:
    train_set = pd.concat([train_set, valid_set])

  train_dir = path.join(out_dir, 'train')
  test_dir = path.join(out_dir, 'test')
  _to_lib_once(train_set, train_dir)
  _to_lib_once(test_set, test_dir)

def to_lib_many():
  to_lib_once(False)
  to_lib_once(True)

def marginalize(data_set):
  n_rating = data_set.rating.unique().shape[0]
  marginal = np.zeros((n_rating))
  for rating in data_set.rating:
    marginal[rating - 1] += 1
  marginal = marginal / marginal.sum()
  return marginal

def engr_cont_feats(train_set, n_cont_feat):
  '''Continuous feature engineering
  item and user features
    0: number of ratings
    1: number of rating=1
    ...
    5: number of rating=5
    6: average rating
  '''
  n_user = train_set.user.unique().shape[0]
  n_item = train_set.item.unique().shape[0]
  user_cont_feats = np.zeros((n_user, n_cont_feat))
  item_cont_feats = np.zeros((n_item, n_cont_feat))
  for row in train_set.itertuples():
    user = row.user
    item = row.item
    rating = row.rating
    user_cont_feats[user, 0] += 1
    user_cont_feats[user, rating] += 1
    user_cont_feats[user, -1] += rating
    item_cont_feats[item, 0] += 1
    item_cont_feats[item, rating] += 1
    item_cont_feats[item, -1] += rating
  for user in range(n_user):
    user_cont_feats[user, -1] /= user_cont_feats[user, 0]
  for item in range(n_item):
    item_cont_feats[item, -1] /= item_cont_feats[item, 0]
  return user_cont_feats, item_cont_feats

def to_resp_once(inc_valid):
  def _to_resp_once(data_set, file_base):
    data_file = file_base + '.dta'
    weight_file = file_base + '.wt'
    is_first = True
    disc_indexes = []
    cont_indexes = []
    with open(data_file, 'w') as fdta, \
        open(weight_file, 'w') as fwt:
      for row_id, row in enumerate(data_set.itertuples()):
        if is_first:
          index = 0
          disc_indexes.append(index)
        user = user_ids[row.user]
        item = item_ids[row.item]
        fdta.write('%d\t%d' % (user, item))
        if is_first:
          index += 2
          disc_indexes.append(index)
        fdta.write('\t%d' % (row_id))
        index += 1
        if is_first:
          disc_indexes.append(index)
          cont_indexes.append(index)
        for i in range(n_cont_feat):
          user_cont_feat = user_cont_feats[row.user][i]
          item_cont_feat = item_cont_feats[row.item][i]
          fdta.write('\t%f' % (user_cont_feat))
          fdta.write('\t%f' % (item_cont_feat))
          if is_first:
            index += 2
            if (i < 1) or (i > 4):
              cont_indexes.append(index)
        fdta.write('\t%d\n' % (row.rating))
        if is_first:
          index += 1
          cont_indexes.append(index)
        is_first = False
        fwt.write('%f\n' % (weights[row.rating - 1]))
    index_file = file_base + '.ind'
    with open(index_file, 'w') as find:
      find.write('disc.')
      for index in disc_indexes:
        find.write('\t%d' % (index))
      find.write('\n')
      find.write('cont.')
      for index in cont_indexes:
        find.write('\t%d' % (index))
      find.write('\n')
    n_rating = len(data_set.index)
    print('Save %d ratings to %s' % (n_rating, data_file))

  base_dir = path.expanduser('~/Downloads/data')
  dir_name = 'music'
  dir_name += '_incl' if inc_valid else '_excl'
  dir_name += '_%s' % (stringify(dflt_ratio))
  out_dir = path.join(base_dir, dir_name)
  if not path.exists(out_dir):
    os.makedirs(out_dir)

  train_set, valid_set, test_set = load_data_sets(dflt_ratio)

  n_user = train_set.user.unique().shape[0]
  n_item = train_set.item.unique().shape[0]
  p_obs = len(train_set.index) / (n_user * n_item)
  p_train = marginalize(train_set)
  p_valid = marginalize(valid_set)
  weights = 1.0 / (p_obs * p_train / p_valid)

  if inc_valid:
    train_set = pd.concat([train_set, valid_set])

  n_cont_feat = 7
  user_cont_feats, item_cont_feats = engr_cont_feats(train_set, n_cont_feat)

  global_id = 0
  users = sorted(train_set.user.unique())
  user_ids = dict()
  for user in users:
    user_ids[user] = global_id
    global_id += 1
  items = sorted(train_set.item.unique())
  item_ids = dict()
  for item in items:
    item_ids[item] = global_id
    global_id += 1

  train_base = path.join(out_dir, 'train')
  valid_base = path.join(out_dir, 'valid')
  test_base = path.join(out_dir, 'test')
  _to_resp_once(train_set, train_base)
  _to_resp_once(valid_set, valid_base)
  _to_resp_once(test_set, test_base)

  user_file = path.join(out_dir, 'user.ft')
  with open(user_file, 'w') as fout:
    for user in users:
      fout.write('%d\n' % (user_ids[user]))
  item_file = path.join(out_dir, 'item.ft')
  with open(item_file, 'w') as fout:
    for item in items:
      fout.write('%d\n' % (item_ids[item]))

def to_resp_many():
  to_resp_once(False)
  to_resp_once(True)

def to_size_once(data_sets, valid_ratio):
  def _to_size_once(data_set, file_base):
    data_file = file_base + '.dta'
    weight_file = file_base + '.wt'
    is_first = True
    disc_indexes = []
    cont_indexes = []
    with open(data_file, 'w') as fdta, \
        open(weight_file, 'w') as fwt:
      for row_id, row in enumerate(data_set.itertuples()):
        if is_first:
          index = 0
          disc_indexes.append(index)
        user = user_ids[row.user]
        item = item_ids[row.item]
        fdta.write('%d\t%d' % (user, item))
        if is_first:
          index += 2
          disc_indexes.append(index)
        fdta.write('\t%d' % (row_id))
        index += 1
        if is_first:
          disc_indexes.append(index)
          cont_indexes.append(index)
        for i in range(n_cont_feat):
          user_cont_feat = user_cont_feats[row.user][i]
          item_cont_feat = item_cont_feats[row.item][i]
          fdta.write('\t%f' % (user_cont_feat))
          fdta.write('\t%f' % (item_cont_feat))
          if is_first:
            index += 2
            if (i < 1) or (i > 4):
              cont_indexes.append(index)
        fdta.write('\t%d\n' % (row.rating))
        if is_first:
          index += 1
          cont_indexes.append(index)
        is_first = False
        fwt.write('%f\n' % (weights[row.rating - 1]))
    index_file = file_base + '.ind'
    with open(index_file, 'w') as find:
      find.write('disc.')
      for index in disc_indexes:
        find.write('\t%d' % (index))
      find.write('\n')
      find.write('cont.')
      for index in cont_indexes:
        find.write('\t%d' % (index))
      find.write('\n')
    n_rating = len(data_set.index)
    print('Save %d ratings to %s' % (n_rating, data_file))

  base_dir = path.expanduser('~/Downloads/data')
  dir_name = 'music'
  dir_name += '_%s' % (stringify(valid_ratio))
  dir_name += '_%s' % (stringify(max_ratio))
  out_dir = path.join(base_dir, dir_name)
  if not path.exists(out_dir):
    os.makedirs(out_dir)

  train_set, valid_set, test_set = data_sets

  n_user = train_set.user.unique().shape[0]
  n_item = train_set.item.unique().shape[0]
  p_obs = len(train_set.index) / (n_user * n_item)
  p_train = marginalize(train_set)
  p_valid = marginalize(valid_set)
  weights = 1.0 / (p_obs * p_train / p_valid)

  n_valid = len(valid_set.index)
  n_valid = int(n_valid * valid_ratio / max_ratio)
  valid_set = valid_set[:n_valid]
  # add this validation set into training set
  train_set = pd.concat([train_set, valid_set])
  n_train = len(train_set.index)
  n_valid = len(valid_set.index)
  n_test = len(test_set.index)
  print('#train=%d #valid=%d #test=%d' % (n_train, n_valid, n_test))

  n_cont_feat = 7
  user_cont_feats, item_cont_feats = engr_cont_feats(train_set, n_cont_feat)

  global_id = 0
  users = sorted(train_set.user.unique())
  user_ids = dict()
  for user in users:
    user_ids[user] = global_id
    global_id += 1
  items = sorted(train_set.item.unique())
  item_ids = dict()
  for item in items:
    item_ids[item] = global_id
    global_id += 1

  train_base = path.join(out_dir, 'train')
  valid_base = path.join(out_dir, 'valid')
  test_base = path.join(out_dir, 'test')
  _to_size_once(train_set, train_base)
  _to_size_once(valid_set, valid_base)
  _to_size_once(test_set, test_base)

def to_size_many():
  data_sets = load_data_sets(max_ratio)
  # ratio_list = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1] + [max_ratio]
  ratio_list = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4] + [max_ratio]
  assert ratio_list[-1] == max_ratio
  for valid_ratio in ratio_list:
    to_size_once(data_sets, valid_ratio)

def main():
  in_dir = path.expanduser('~/Downloads/data/Webscope_R3')
  if not path.exists(in_dir):
    raise Exception('Please download the music dataset from Yahoo.')
  shuffle_data(in_dir)

  choices = ['lib', 'resp', 'size']
  parser = argparse.ArgumentParser()
  parser.add_argument('out_format', choices=choices)
  args = parser.parse_args()
  out_format = args.out_format
  if out_format == choices[0]:
    to_lib_many()
  if out_format == choices[1]:
    to_resp_many()
  if out_format == choices[2]:
    to_size_many()

if __name__ == '__main__':
  main()

