from os import path

import argparse
import numpy as np
import os

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

def load_dataset(in_file):
  dense_data = np.loadtxt(in_file, dtype=np.int32)
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

def shuffle_data(in_dir, data_dir):
  if path.exists(data_dir):
    return
  biased_file = path.join(in_dir, 'train.ascii')
  unbiased_file = path.join(in_dir, 'test.ascii')
  n_user, n_item, biased_data = load_dataset(biased_file)
  n_user, n_item, unbiased_data = load_dataset(unbiased_file)
  print('#user=%d #item=%d' % (n_user, n_item))
  n_biased = len(biased_data)
  n_unbiased = len(unbiased_data)
  print('#biased=%d #unbiased=%d' % (n_biased, n_unbiased))
  np.random.seed(0)
  np.random.shuffle(biased_data)
  np.random.shuffle(unbiased_data)
  os.makedirs(data_dir)
  biased_file = path.join(data_dir, 'biased.dta')
  unbiased_file = path.join(data_dir, 'unbiased.dta')
  save_dataset(biased_data, biased_file)
  save_dataset(unbiased_data, unbiased_file)

def main():
  in_dir = path.expanduser('~/Downloads/data/coat')
  maybe_download(in_dir)
  data_dir = 'data'
  shuffle_data(in_dir, data_dir)

  parser = argparse.ArgumentParser()
  parser.add_argument('out_format', choices=['lib', 'resp'])
  parser.add_argument('ubs_ratio', type=float)
  args = parser.parse_args()
  out_format = args.out_format
  ubs_ratio = args.ubs_ratio
  if out_format == 'lib':
    out_dir = path.expanduser('')

if __name__ == '__main__':
  main()

