from os import path

import argparse
import math
import numpy as np

temp_dir = "~/Downloads/data/coat"
temp_dir = path.expanduser(temp_dir)
prop_file = path.join(temp_dir, "propensities.ascii")

data_dir = "coat"
lower_file = path.join(data_dir, "lower.txt")
upper_file = path.join(data_dir, "upper.txt")
valid_file = path.join(data_dir, "valid.txt")
test_file = path.join(data_dir, "test.txt")
n_rating = 5

def main():
  if not path.exists(temp_dir):
    print("Download from https://www.cs.cornell.edu/~schnabts/mnar/coat.zip")
    exit()

  lower_set = np.loadtxt(lower_file, dtype=np.int32)
  valid_set = np.loadtxt(valid_file, dtype=np.int32)
  biased_set = np.concatenate([lower_set, valid_set], axis=0)
  n_biased = len(biased_set)
  print("Find %d biased ratings" % n_biased)

  test_set = np.loadtxt(test_file, dtype=np.int32)
  n_test = len(test_set)
  print("Find %d test ratings" % n_test)

  n_user = len(np.unique(biased_set[:, 0]))
  n_item = len(np.unique(biased_set[:, 1]))
  print("Find %d users and %d items" % (n_user, n_item))

  p_mat = np.loadtxt(prop_file, dtype=np.float32)
  mult = 0
  for user, item, _ in biased_set:
    mult += (1 / p_mat[user, item])
  mult /= (n_user * n_item)
  print("Multiply propensities by %.6f" % mult)
  p_mat *= mult
  mult = 0
  for user, item, _ in biased_set:
    mult += (1 / p_mat[user, item])
  print("Inverse propensity sum is %.6f" % mult)

  test_p = []
  for user, item, _ in test_set:
    test_p.append(p_mat[user, item])
  sample_var = np.var(test_p, ddof=1)
  inv_square = 0
  for user, item, _ in test_set:
    inv_square += (1 / pow(p_mat[user, item], 2))
  inv_square /= n_test
  print("Sample variance is %s" % "{:.3E}".format(sample_var))
  print("Inverse square is %s" % "{:.3E}".format(inv_square))

if __name__ == '__main__':
  main()



