from os import path

import numpy as np
import os
import random

class Dataset(object):
  def __init__(self, arg):
    self.arg = arg

def maybe_download(data_dir):
  coat_url = 'https://www.cs.cornell.edu/~schnabts/mnar/coat.zip'

  if path.exists(data_dir):
    return

  data_dir = path.dirname(data_dir)
  if not path.exists(data_dir):
    os.makedirs(data_dir)

  coat_zip = path.join(data_dir, 'data.zip')
  os.system('wget %s -O %s' % (coat_url, coat_zip))
  os.system('unzip %s -d %s' % (coat_zip, data_dir))
  os.system('rm -f %s' % (coat_zip))

def load_dataset(in_file):
  dns_ratings = np.loadtxt(in_file, dtype=np.int32)
  n_user, n_item = dns_ratings.shape

  coo_ratings = []
  users, items = dns_ratings.nonzero()
  for user, item in zip(users, items):
    rating = dns_ratings[user, item]
    item += n_user # crucial
    coo_ratings.append((user, item, rating))
  return n_user, n_item, coo_ratings

def save_dataset(ratings, out_file):
  n_rating = len(ratings)
  with open(out_file, 'w') as fout:
    for user, item, rating in ratings:
      fout.write('%d\t%d\t%d\n' % (rating, user, item))
  print('save %d ratings to %s' % (n_rating, out_file))

def split_dataset(data_dir):
  biased_file = path.join(data_dir, 'train.ascii')
  unbiased_file = path.join(data_dir, 'test.ascii')
  n_user, n_item, biased_ratings = load_dataset(biased_file)
  n_user, n_item, unbiased_ratings = load_dataset(unbiased_file)
  print('#user=%d #item=%d' % (n_user, n_item))

  n_biased = len(biased_ratings)
  n_unbiased = len(unbiased_ratings)
  print('#biased=%d #unbiased=%d' % (n_biased, n_unbiased))

  valid_ratio = 0.10
  train_ratings = biased_ratings
  random.shuffle(unbiased_ratings)
  n_valid = int(valid_ratio * n_unbiased)
  valid_ratings = unbiased_ratings[:n_valid]
  test_ratings = unbiased_ratings[n_valid:]
  valid_ratings = unbiased_ratings
  test_ratings = unbiased_ratings

  n_train = len(train_ratings)
  n_valid = len(valid_ratings)
  n_test = len(test_ratings)
  print('#train=%d #valid=%d #test=%d' % (n_train, n_valid, n_test))

  train_file = path.join(data_dir, 'train.dta')
  valid_file = path.join(data_dir, 'valid.dta')
  test_file = path.join(data_dir, 'test.dta')
  save_dataset(train_ratings, train_file)
  save_dataset(valid_ratings, valid_file)
  save_dataset(test_ratings, test_file)

def main():
  data_dir = path.expanduser('~/Downloads/data/coat')
  maybe_download(data_dir)
  split_dataset(data_dir)

if __name__ == '__main__':
  main()

