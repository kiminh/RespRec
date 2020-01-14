import math
import os
import shutil

import numpy as np

from os import path

temp_dir = "~/Downloads/data/Webscope_R3"
temp_dir = path.expanduser(temp_dir)

data_dir = "music"
if not path.exists(data_dir):
  os.makedirs(data_dir)
lower_file = path.join(data_dir, "lower.txt")
upper_file = path.join(data_dir, "upper.txt")
valid_file = path.join(data_dir, "valid.txt")
test_file = path.join(data_dir, "test.txt")

def load_data_set(in_file):
  coo_data = np.loadtxt(in_file, dtype=np.int32)
  coo_data[:, :2] -= 1
  n_user = len(np.unique(coo_data[:, 0]))
  n_item = len(np.unique(coo_data[:, 1]))
  print("Find %d users and %d items in %s" % (n_user, n_item, in_file))
  return coo_data

def save_data_set(data_set, out_file):
  out_dir = path.dirname(out_file)
  if not path.exists(out_dir):
    os.makedirs(out_dir)

  n_rating = len(data_set)
  with open(out_file, "w") as fout:
    for user, item, rating in data_set:
      fout.write("%d\t%d\t%d\n" % (user, item, rating))
  print("Save %d ratings into %s" % (n_rating, out_file))

def split_data():
  if not path.exists(temp_dir):
    print("Please download the music dataset from Yahoo")
    exit()

  biased_file = path.join(temp_dir, 'ydata-ymusic-rating-study-v1_0-train.txt')
  unbiased_file = path.join(temp_dir, 'ydata-ymusic-rating-study-v1_0-test.txt')
  biased_set = load_data_set(biased_file)
  unbiased_set = load_data_set(unbiased_file)
  n_biased = len(biased_set)
  n_unbiased = len(unbiased_set)
  print("Find %d biased and %d unbiased ratings" % (n_biased, n_unbiased))

  min_user = biased_set[:, 0].min()
  max_user = biased_set[:, 0].max()
  print("User ID ranges from %d to %d" % (min_user, max_user))
  min_item = biased_set[:, 1].min()
  max_item = biased_set[:, 1].max()
  print("Item ID ranges from %d to %d" % (min_item, max_item))

  np.random.seed(19931201)
  np.random.shuffle(biased_set)
  np.random.shuffle(unbiased_set)
  n_lower = math.ceil(.9 * n_biased)
  n_upper = math.ceil(.05 * n_unbiased)
  # Lower train set
  save_data_set(biased_set[:n_lower], lower_file)
  # Upper train set
  save_data_set(unbiased_set[:n_upper], upper_file)
  # Validation set
  save_data_set(biased_set[n_lower:], valid_file)
  # Test set
  save_data_set(unbiased_set[n_upper:], test_file)

def load_data_sets():
  lower_set = np.loadtxt(lower_file, dtype=np.int32)
  upper_set = np.loadtxt(upper_file, dtype=np.int32)
  train_set = np.concatenate([lower_set, upper_set], axis=0)
  valid_set = np.loadtxt(valid_file, dtype=np.int32)
  test_set = np.loadtxt(test_file, dtype=np.int32)
  n_train, _ = train_set.shape
  n_valid, _ = valid_set.shape
  n_test, _ = test_set.shape
  print("Load %d train, %d valid, and %d test ratings" % (n_train, n_valid, n_test))
  return train_set, valid_set, test_set

def for_librec():
  train_set, valid_set, test_set = load_data_sets()

  out_dir = path.expanduser('~/Projects/librec/data')
  out_dir = path.join(out_dir, "beta_music")
  if path.exists(out_dir) and path.isdir(out_dir):
      shutil.rmtree(out_dir)
  
  train_file = path.join(out_dir, "train", "ratings.txt")
  save_data_set(train_set, train_file)
  test_file = path.join(out_dir, "test", "ratings.txt")
  save_data_set(test_set, test_file)

def main():
  split_data()
  for_librec()

if __name__ == '__main__':
  main()