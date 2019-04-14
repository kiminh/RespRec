from os import path
from sklearn import utils

import math
import os
import pandas as pd

dnld_dir = path.expanduser('~/Downloads')
in_dir = path.join(dnld_dir, 'ml-100k')
in_rating_file = path.join(in_dir, 'u.data')
in_user_file = path.join(in_dir, 'u.user')

cur_dir = path.dirname(path.realpath(__file__))
dataset = path.basename(__file__).split('.')[0]
out_dir = path.join(cur_dir, dataset)
if not path.exists(out_dir):
  os.makedirs(out_dir)
out_train_file = path.join(out_dir, 'train.data')
out_test_file = path.join(out_dir, 'test.data')
out_stat_file = path.join(out_dir, 'data.stat')
out_user_file = path.join(out_dir, 'user.attr')

all_ratings = pd.read_csv(in_rating_file, sep='\t', names=['u', 'i', 'r', 't'])
# all_ratings = all_ratings.sort_values(['t', 'i'])
all_ratings = utils.shuffle(all_ratings)
all_ratings.u = all_ratings.u - 1
all_ratings.i = all_ratings.i - 1
num_users = all_ratings.u.unique().shape[0]
num_items = all_ratings.i.unique().shape[0]

num_ratings = all_ratings.shape[0]
train_size = math.ceil(0.8 * num_ratings)
kwargs = {'sep': '\t', 'header': False, 'index':False}
train_ratings = all_ratings[:train_size]
train_ratings.to_csv(out_train_file, **kwargs)
test_ratings = all_ratings[train_size:]
test_ratings.to_csv(out_test_file, **kwargs)
print(num_users, train_ratings.u.unique().shape, test_ratings.u.unique().shape)
print(num_items, train_ratings.i.unique().shape, test_ratings.i.unique().shape)
train_size = train_ratings.shape[0]
test_size = test_ratings.shape[0]

with open(out_stat_file, 'w') as fout:
  fout.write('#users=%d\n' % (num_users))
  fout.write('#items=%d\n' % (num_items))
  fout.write('#train=%d\n' % (train_size))
  fout.write('#test=%d\n' % (test_size))


user_info = pd.read_csv(in_user_file, sep='|', names=['u', 'a', 'g', 'o', 'c'])
user_info.u = user_info.u - 1
kwargs['columns'] = ['u', 'g']
user_info.to_csv(out_user_file, **kwargs)



