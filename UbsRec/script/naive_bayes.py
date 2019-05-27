from os import path

import argparse
import pandas as pd

def marginalize(ratings):
  marginal = dict()
  for row in ratings.itertuples():
    rating = row.rating
    marginal[rating] = marginal.get(rating, 0.) + 1.
  normalizer = sum(marginal.values())
  marginal = {k: v / normalizer for k, v in marginal.items()}
  for rating in sorted(marginal.keys()):
    print('%d:%.4f' % (rating, marginal[rating]), end=' ')
  print('')
  return marginal

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('data_dir', type=str)
  args = parser.parse_args()

  data_dir = args.data_dir
  train_file = path.join(data_dir, 'train.dta')
  valid_file = path.join(data_dir, 'valid.dta')
  test_file = path.join(data_dir, 'test.dta')

  read_kwargs = {'sep': '\t', 'names': ['user', 'item', 'rating']}
  train_ratings = pd.read_csv(train_file, **read_kwargs)
  train_marginal = marginalize(train_ratings)
  valid_ratings = pd.read_csv(valid_file, **read_kwargs)
  valid_marginal = marginalize(valid_ratings)
  test_ratings = pd.read_csv(test_file, **read_kwargs)
  test_marginal = marginalize(test_ratings)

if __name__ == '__main__':
  main()



