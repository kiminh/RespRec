from os import path

import argparse
import pandas as pd

import read_data

def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('data_dir', type=str)
  parser.add_argument('result_file', type=str)
  return parser.parse_args()

if __name__ == '__main__':
  args = parse_args()
  num_users, num_items, test_data = read_data.read_test(args.data_dir)
  user_file = path.join(args.data_dir, 'user.info')
  user_info = pd.read_csv(user_file, sep='\t', names=['u', 'g'])
  user_info = user_info.set_index('u').to_dict()['g']

  

