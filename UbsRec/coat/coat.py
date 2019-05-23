'''
coat
  +vd: add validation set into training set
  +ft: use user and item features for training
'''
from os import path

import argparse
import math
import numpy as np
import os
import pandas as pd

class Dataset(object):
  def __init__(self, arg):
    self.arg = arg

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

def load_dataset(data_file):
  dense_data = np.loadtxt(data_file, dtype=np.int32)
  n_user, n_item = dense_data.shape
  coo_data = []
  users, items = dense_data.nonzero()
  for user, item in zip(users, items):
    rating = dense_data[user, item]
    coo_data.append((user, item, rating))
  return n_user, n_item, coo_data

def save_dataset(ratings, out_file):
  n_rating = len(ratings)
  with open(out_file, 'w') as fout:
    for user, item, rating in ratings:
      fout.write('%d\t%d\t%d\n' % (user, item, rating))
  print('save %d ratings to %s' % (n_rating, out_file))

def shuffle_data(in_dir):
  data_dir = 'data'
  if path.exists(data_dir):
    return
  biased_file = path.join(in_dir, 'train.ascii')
  unbiased_file = path.join(in_dir, 'test.ascii')
  n_user, n_item, biased_set = load_dataset(biased_file)
  n_user, n_item, unbiased_set = load_dataset(unbiased_file)
  print('#user=%d #item=%d' % (n_user, n_item))
  n_biased = len(biased_set)
  n_unbiased = len(unbiased_set)
  print('#biased=%d #unbiased=%d' % (n_biased, n_unbiased))
  np.random.seed(0)
  np.random.shuffle(biased_set)
  np.random.shuffle(unbiased_set)
  os.makedirs(data_dir)
  biased_file = path.join(data_dir, 'biased.dta')
  unbiased_file = path.join(data_dir, 'unbiased.dta')
  save_dataset(biased_set, biased_file)
  save_dataset(unbiased_set, unbiased_file)

def load_datasets(ubs_ratio):
  data_dir = 'data'
  biased_file = path.join(data_dir, 'biased.dta')
  unbiased_file = path.join(data_dir, 'unbiased.dta')
  names = ['user', 'item', 'rating']
  biased_set = pd.read_csv(biased_file, sep='\t', names=names)
  unbiased_set = pd.read_csv(unbiased_file, sep='\t', names=names)

  train_set = biased_set
  n_unbiased = len(unbiased_set.index)
  n_valid = math.ceil(ubs_ratio * n_unbiased)
  valid_set = unbiased_set[:n_valid]
  test_set = unbiased_set[n_valid:]
  return train_set, valid_set, test_set

def stringify(number):
  string = '%f' % (number)
  string = string.rstrip('0')
  string = string[:-1] if string.endswith('.') else string
  return string

def to_lib_once(ubs_ratio, plus_vd):
  def _to_lib_once(ratings, out_dir):
    kwargs = {'sep': '\t', 'header': False, 'index':False}
    os.makedirs(out_dir)
    out_file = path.join(out_dir, 'ratings.txt')
    n_rating = len(ratings.index)
    print('Save %d ratings to %s' % (n_rating, out_file))
    ratings.to_csv(out_file, **kwargs)

  base_dir = path.expanduser('~/Projects/librec/data')
  dir_name = 'coat'
  dir_name += '+vd' if plus_vd else '-vd'
  dir_name += '_%s' % (stringify(ubs_ratio))
  out_dir = path.join(base_dir, dir_name)
  if path.exists(out_dir):
    return
  os.makedirs(out_dir)

  train_set, valid_set, test_set = load_datasets(ubs_ratio)
  if plus_vd:
    train_set = pd.concat([train_set, valid_set])

  train_dir = path.join(out_dir, 'train')
  test_dir = path.join(out_dir, 'test')
  _to_lib_once(train_set, train_dir)
  _to_lib_once(test_set, test_dir)

def to_lib_many(ubs_ratio):
  to_lib_once(ubs_ratio, False)
  to_lib_once(ubs_ratio, True)

def load_feature(feat_file):
  features = []
  dense_data = np.loadtxt(feat_file, dtype=np.int32)
  for i in range(dense_data.shape[0]):
    feature = dense_data[i].nonzero()[0]
    assert len(feature) == 4
    features.append(feature)
  features = np.asarray(features)
  return features

def load_features():
  in_dir = path.expanduser('~/Downloads/data/coat')
  feat_dir = path.join(in_dir, 'user_item_features')
  user_file = path.join(feat_dir, 'user_features.ascii')
  item_file = path.join(feat_dir, 'item_features.ascii')
  user_features = load_feature(user_file)
  item_features = load_feature(item_file)
  return user_features, item_features

def to_coat_once(ubs_ratio, plus_vd, plus_ft):
  def _to_coat_once(ratings, out_file):
    with open(out_file, 'w') as fout:
      for row in ratings.itertuples():
        uid = uid_gid[row.user]
        iid = iid_gid[row.item]
        fout.write('%d\t%d' % (uid, iid))
        if plus_ft:
          for ufid in user_features[row.user]:
            fout.write('\t%d' % (ufid_gid[ufid]))
          for ifid in item_features[row.item]:
            fout.write('\t%d' % (ifid_gid[ifid]))
        fout.write('\t%d\n' % (row.rating))
    n_rating = len(ratings.index)
    print('save %d ratings to %s' % (n_rating, out_file))

  base_dir = path.expanduser('~/Downloads/data')
  dir_name = 'coat'
  dir_name += '+vd' if plus_vd else '-vd'
  dir_name += '+ft' if plus_ft else '-ft'
  dir_name += '_%s' % (stringify(ubs_ratio))
  out_dir = path.join(base_dir, dir_name)
  if path.exists(out_dir):
    return
  os.makedirs(out_dir)

  train_set, valid_set, test_set = load_datasets(ubs_ratio)
  if plus_vd:
    train_set = pd.concat([train_set, valid_set])
  user_features, item_features = load_features()

  gid = 0
  uid_gid = dict()
  for uid in sorted(train_set.user.unique()):
    uid_gid[uid] = gid
    gid += 1
  iid_gid = dict()
  for iid in sorted(train_set.item.unique()):
    iid_gid[iid] = gid
    gid += 1
  ufid_gid = dict()
  for ufid in sorted(np.unique(user_features)):
    ufid_gid[ufid] = gid
    gid += 1
  ifid_gid = dict()
  for ifid in sorted(np.unique(item_features)):
    ifid_gid[ifid] = gid
    gid += 1

  train_file = path.join(out_dir, 'train.dta')
  valid_file = path.join(out_dir, 'valid.dta')
  test_file = path.join(out_dir, 'test.dta')
  _to_coat_once(train_set, train_file)
  _to_coat_once(valid_set, valid_file)
  _to_coat_once(test_set, test_file)

def to_coat_many(ubs_ratio):
  to_coat_once(ubs_ratio, False, False)
  to_coat_once(ubs_ratio, False, True)
  to_coat_once(ubs_ratio, True, False)
  to_coat_once(ubs_ratio, True, True)

def main():
  in_dir = path.expanduser('~/Downloads/data/coat')
  maybe_download(in_dir)
  shuffle_data(in_dir)

  parser = argparse.ArgumentParser()
  parser.add_argument('out_format', choices=['lib', 'coat'])
  parser.add_argument('ubs_ratio', type=float)
  args = parser.parse_args()
  out_format = args.out_format
  ubs_ratio = args.ubs_ratio
  if out_format == 'lib':
    to_lib_many(ubs_ratio)
  if out_format == 'coat':
    to_coat_many(ubs_ratio)

if __name__ == '__main__':
  main()

