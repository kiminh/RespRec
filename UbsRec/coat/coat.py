from os import path

import argparse
import numpy as np
import os

class Dataset(object):
  def __init__(self, arg):
    self.arg = arg

def maybe_download(in_dir):
  coat_url = 'https://www.cs.cornell.edu/~schnabts/mnar/coat.zip'

  if path.exists(in_dir):
    return

  in_dir = path.dirname(in_dir)
  if not path.exists(in_dir):
    os.makedirs(in_dir)

  coat_zip = path.join(in_dir, 'data.zip')
  os.system('wget %s -O %s' % (coat_url, coat_zip))
  os.system('unzip %s -d %s' % (coat_zip, in_dir))
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

def split_dataset(in_dir, out_dir):
  biased_file = path.join(in_dir, 'train.ascii')
  unbiased_file = path.join(in_dir, 'test.ascii')
  n_user, n_item, biased_ratings = load_dataset(biased_file)
  n_user, n_item, unbiased_ratings = load_dataset(unbiased_file)
  print('#user=%d #item=%d' % (n_user, n_item))

  n_biased = len(biased_ratings)
  n_unbiased = len(unbiased_ratings)
  print('#biased=%d #unbiased=%d' % (n_biased, n_unbiased))

  valid_ratio = 0.1
  train_ratings = biased_ratings
  np.random.seed(0)
  np.random.shuffle(unbiased_ratings)
  n_valid = int(valid_ratio * n_unbiased)
  valid_ratings = unbiased_ratings[:n_valid]
  test_ratings = unbiased_ratings[n_valid:]

  n_train = len(train_ratings)
  n_valid = len(valid_ratings)
  n_test = len(test_ratings)
  print('#train=%d #valid=%d #test=%d' % (n_train, n_valid, n_test))

  if not path.exists(out_dir):
    os.makedirs(out_dir)
  train_file = path.join(out_dir, 'train.dta')
  valid_file = path.join(out_dir, 'valid.dta')
  test_file = path.join(out_dir, 'test.dta')
  save_dataset(train_ratings, train_file)
  save_dataset(valid_ratings, valid_file)
  save_dataset(test_ratings, test_file)

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('out_format', type=str)
  parser.add_argument('ubs_ratio', type=float)
  args = parser.parse_args()

  in_dir = path.expanduser('~/Downloads/data/coat')
  maybe_download(in_dir)
  out_dir = 'data'
  split_dataset(in_dir, out_dir)

if __name__ == '__main__':
  main()

