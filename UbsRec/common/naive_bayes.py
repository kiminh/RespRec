from os import path

import argparse
import pandas as pd

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('data_dir', type=str)
  args = parser.parse_args()

  data_dir = args.data_dir
  train_file = path.join(data_dir, 'train.dta')
  valid_file = path.join(data_dir, 'valid.dta')
  test_file = path.join(data_dir, 'test.dta')

  read_kwargs = {'sep': '\t', 'names': ['user', 'item', 'rating']}
  train_ratings = pd.read_csv(biased_file, **read_kwargs)


if __name__ == '__main__':
  main()