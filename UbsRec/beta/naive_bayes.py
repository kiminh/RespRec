from os import path

import argparse
import math
import numpy as np

data_dir = "music"
lower_file = path.join(data_dir, "lower.txt")
upper_file = path.join(data_dir, "upper.txt")
valid_file = path.join(data_dir, "valid.txt")
test_file = path.join(data_dir, "test.txt")
n_rating = 5

def load_data_sets():
  lower_set = np.loadtxt(lower_file, dtype=np.int32)
  valid_set = np.loadtxt(valid_file, dtype=np.int32)
  biased_set = np.concatenate([lower_set, valid_set], axis=0)

  unbiased_set = np.loadtxt(upper_file, dtype=np.int32)

  test_set = np.loadtxt(test_file, dtype=np.int32)

  return biased_set, unbiased_set, test_set

def marginalize(data_set):
  marginal = np.zeros((n_rating))
  for rating in data_set[:, 2]:
    marginal[rating - 1] += 1
  marginal = marginal / marginal.sum()
  # for rating in range(n_rating):
  #   print("%d:%.4f" % (rating, marginal[rating]), end=" ")
  # print("")
  return marginal

def main():
  biased_set, unbiased_set, test_set = load_data_sets()

  n_biased = len(biased_set)
  n_unbiased = len(unbiased_set)
  print("Find %d biased and %d unbiased ratings" % (n_biased, n_unbiased))

  biased_marginal = marginalize(biased_set)
  unbiased_marginal = marginalize(unbiased_set)

  n_user = len(np.unique(biased_set[:, 0]))
  n_item = len(np.unique(biased_set[:, 1]))
  p_obs = n_biased / (n_user * n_item)
  p_arr = p_obs * biased_marginal / unbiased_marginal
  for rating in range(n_rating):
    print("%f" % (p_arr[rating]), end=" ")
  print("\t")

  mult = 0
  for rating in biased_set[:, 2]:
    mult += (1 / p_arr[rating - 1])
  mult /= (n_user * n_item)
  print("Multiply propensities by %.6f" % mult)

  test_p = []
  for rating in test_set[:, 2]:
    test_p.append(p_arr[rating - 1])
  sample_var = np.var(test_p, ddof=1)
  inv_square = 0
  n_test = len(test_p)
  print("Find %d test ratings" % n_test)
  for rating in test_set[:, 2]:
    inv_square += (1 / pow(p_arr[rating - 1], 2))
  inv_square /= n_test
  print("Sample variance is %s" % "{:.3E}".format(sample_var))
  print("Inverse square is %s" % "{:.3E}".format(inv_square))

if __name__ == '__main__':
  main()



