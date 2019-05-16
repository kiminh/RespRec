from os import path

import argparse

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('data_dir', type=str)
  args = parser.parse_args()

  data_dir = args.data_dir
  train_file = path.join(data_dir, 'train.dta')
  valid_file = path.join(data_dir, 'valid.dta')
  test_file = path.join(data_dir, 'test.dta')

if __name__ == '__main__':
  main()