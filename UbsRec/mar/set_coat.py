'''Separate a small portion of unbiased data as the valid set
excl/incl: exclude/include the valid set from/into the train set
'''
from os import path

import argparse
import math
import numpy as np
import os
import pandas as pd

valid_ratio = 0.05

def maybe_download(in_dir):
  if path.exists(in_dir):
    return
  in_dir = path.dirname(in_dir)
  if not path.exists(in_dir):
    os.makedirs(in_dir)
  coat_url = 'https://www.cs.cornell.edu/~schnabts/mnar/coat.zip'
  coat_zip = path.join(in_dir, 'data.zip')
  os.system('wget %s -O %s' % (coat_url, coat_zip))
  os.system('unzip %s -d %s' % (coat_zip, in_dir))
  os.system('rm -f %s' % (coat_zip))

def load_data_set(data_file):
  dense_data = np.loadtxt(data_file, dtype=np.int32)
  n_user, n_item = dense_data.shape
  coo_data = []
  users, items = dense_data.nonzero()
  for user, item in zip(users, items):
    rating = dense_data[user, item]
    coo_data.append((user, item, rating))
  return n_user, n_item, coo_data

def save_data_set(data_set, data_file):
  n_rating = len(data_set)
  with open(data_file, 'w') as fout:
    for user, item, rating in data_set:
      fout.write('%d\t%d\t%d\n' % (user, item, rating))
  print('Save %d ratings to %s' % (n_rating, data_file))

def shuffle_data(in_dir):
  data_dir = 'data'
  if not path.exists(data_dir):
    os.makedirs(data_dir)
  biased_file = path.join(in_dir, 'train.ascii')
  unbiased_file = path.join(in_dir, 'test.ascii')
  n_user, n_item, biased_set = load_data_set(biased_file)
  n_user, n_item, unbiased_set = load_data_set(unbiased_file)
  print('#user=%d #item=%d' % (n_user, n_item))
  n_biased = len(biased_set)
  n_unbiased = len(unbiased_set)
  print('#biased=%d #unbiased=%d' % (n_biased, n_unbiased))
  np.random.seed(0)
  np.random.shuffle(biased_set)
  np.random.shuffle(unbiased_set)
  biased_file = path.join(data_dir, 'biased.dta')
  unbiased_file = path.join(data_dir, 'unbiased.dta')
  save_data_set(biased_set, biased_file)
  save_data_set(unbiased_set, unbiased_file)

def load_data_sets():
  data_dir = 'data'
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

def stringify(number):
  string = '%f' % (number)
  string = string.rstrip('0')
  string = string[:-1] if string.endswith('.') else string
  return string

def to_lib_once(inc_valid):
  def _to_lib_once(ratings, out_dir):
    if not path.exists(out_dir):
      os.makedirs(out_dir)
    kwargs = {'sep': '\t', 'header': False, 'index':False}
    data_file = path.join(out_dir, 'ratings.txt')
    n_rating = len(ratings.index)
    ratings.to_csv(data_file, **kwargs)
    print('Save %d ratings to %s' % (n_rating, data_file))

  base_dir = path.expanduser('~/Projects/librec/data')
  dir_name = 'coat'
  dir_name += '_incl' if inc_valid else '_excl'
  dir_name += '_%s' % (stringify(valid_ratio))
  out_dir = path.join(base_dir, dir_name)
  train_set, valid_set, test_set = load_data_sets()
  if inc_valid:
    train_set = pd.concat([train_set, valid_set])

  train_dir = path.join(out_dir, 'train')
  test_dir = path.join(out_dir, 'test')
  _to_lib_once(train_set, train_dir)
  _to_lib_once(test_set, test_dir)

def to_lib_many():
  to_lib_once(False)
  to_lib_once(True)

def load_disc_feat(feat_file):
  disc_feats = []
  dense_data = np.loadtxt(feat_file, dtype=np.int32)
  for i in range(dense_data.shape[0]):
    feature = dense_data[i].nonzero()[0]
    assert len(feature) == 4
    disc_feats.append(feature)
  disc_feats = np.asarray(disc_feats)
  return disc_feats

def load_disc_feats():
  in_dir = path.expanduser('~/Downloads/data/coat')
  feat_dir = path.join(in_dir, 'user_item_features')
  user_file = path.join(feat_dir, 'user_features.ascii')
  item_file = path.join(feat_dir, 'item_features.ascii')
  user_disc_feats = load_disc_feat(user_file)
  item_disc_feats = load_disc_feat(item_file)
  return user_disc_feats, item_disc_feats

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
        if is_first:
          index += 2
          disc_indexes.append(index)
        fdta.write('%d\t%d' % (user, item))
        for user_disc_feat in user_disc_feats[row.user]:
          fdta.write('\t%d' % (user_feat_ids[user_disc_feat]))
          if is_first:
            index += 1
        for item_disc_feat in item_disc_feats[row.item]:
          fdta.write('\t%d' % (item_feat_ids[item_disc_feat]))
          if is_first:
            index += 1
        if is_first:
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

        fwt.write('%f\n' % (weights[row.user, row.item]))

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
  dir_name = 'coat'
  dir_name += '_incl' if inc_valid else '_excl'
  dir_name += '_%s' % (stringify(valid_ratio))
  out_dir = path.join(base_dir, dir_name)
  if not path.exists(out_dir):
    os.makedirs(out_dir)

  train_set, valid_set, test_set = load_data_sets()

  in_dir = path.expanduser('~/Downloads/data/coat')
  weight_file = path.join(in_dir, 'propensities.ascii')
  weights = np.loadtxt(weight_file, dtype=np.float32)

  if inc_valid:
    train_set = pd.concat([train_set, valid_set])
  user_disc_feats, item_disc_feats = load_disc_feats()
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
  user_feat_ids = dict()
  for user_disc_feat in sorted(np.unique(user_disc_feats)):
    user_feat_ids[user_disc_feat] = global_id
    global_id += 1
  item_feat_ids = dict()
  for item_disc_feat in sorted(np.unique(item_disc_feats)):
    item_feat_ids[item_disc_feat] = global_id
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
      assert user == user_ids[user]
      fout.write('%d' % (user_ids[user]))
      for user_disc_feat in user_disc_feats[user]:
        fout.write('\t%d' % (user_feat_ids[user_disc_feat]))
      fout.write('\n')
  item_file = path.join(out_dir, 'item.ft')
  with open(item_file, 'w') as fout:
    for item in items:
      assert item + len(users) == item_ids[item]
      fout.write('%d' % (item_ids[item]))
      for item_disc_feat in item_disc_feats[item]:
        fout.write('\t%d' % (item_feat_ids[item_disc_feat]))
      fout.write('\n')

def to_resp_many():
  to_resp_once(False)
  to_resp_once(True)

def main():
  in_dir = path.expanduser('~/Downloads/data/coat')
  maybe_download(in_dir)
  shuffle_data(in_dir)

  parser = argparse.ArgumentParser()
  parser.add_argument('out_format', choices=['lib', 'resp'])
  args = parser.parse_args()
  out_format = args.out_format
  if out_format == 'lib':
    to_lib_many()
  if out_format == 'resp':
    to_resp_many()

if __name__ == '__main__':
  main()

