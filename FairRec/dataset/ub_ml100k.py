from os import path
from sklearn import utils

import math
import numpy as np
import os
import pandas as pd
import random

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

user_attr = pd.read_csv(in_user_file, sep='|', names=['u', 'a', 'g', 'o', 'c'])
user_attr.u = user_attr.u - 1
user_attr = user_attr.set_index('u').to_dict()['g']

num_m = 0
num_f = 0
for u in all_ratings.u.unique():
  attr = user_attr[u]
  if attr == 'M':
    num_m += 1
  elif attr == 'F':
    num_f += 1
  else:
    raise Exception('unknown attribute %s' % (attr))
print('#male=%d #female=%d' % (num_m, num_f))
exp_m = int(num_f / (num_m / num_f))
print('male num=%d exp=%d' % (num_m, exp_m))

m_nr_dict = {}
f_nr_dict = {}
for row in all_ratings.itertuples():
  attr = user_attr[row.u]
  if attr == 'M':
    m_nr_dict[row.u] = m_nr_dict.get(row.u, 0) + 1
  elif attr == 'F':
    f_nr_dict[row.u] = f_nr_dict.get(row.u, 0) + 1
  else:
    raise Exception('unknown attribute %s' % (attr))
nr_m_dict = {}
nr_f_dict = {}
for m, nr in m_nr_dict.items():
  if nr not in nr_m_dict:
    nr_m_dict[nr] = set()
  nr_m_dict[nr].add(m)
for f, nr in f_nr_dict.items():
  if nr not in nr_f_dict:
    nr_f_dict[nr] = set()
  nr_f_dict[nr].add(f)
m_set = set(m_nr_dict.keys())
assert len(m_set) == num_m
f_set = set(f_nr_dict.keys())
assert len(f_set) == num_f
ttl_f = sum(f_nr_dict.values())
avg_f = ttl_f / num_f
ttl_m = sum(m_nr_dict.values())
avg_m = ttl_m / num_m
print('avg male=%.2f female=%.2f' % (avg_m, avg_f))
pct_m = 0.96 # m > f
pct_m = 0.88 # f > m
avg_m = avg_f * pct_m

sel_m = set()
m_minus_f = 0
min_nr = min(nr_m_dict.keys()) - 1
max_nr = max(nr_m_dict.keys()) + 1
for j in range(exp_m):
  f_nr = avg_f
  f_nr = avg_m
  if m_minus_f < 0:
    can_m = set()
    for m_nr in range(math.ceil(f_nr), max_nr, 1):
      if m_nr in nr_m_dict:
        can_m = can_m.union(nr_m_dict[m_nr])
    rnd_m = random.choice(list(can_m))
    sel_m.add(rnd_m)
    m_nr = m_nr_dict[rnd_m]
    nr_m_dict[m_nr].remove(rnd_m)
    m_minus_f += (m_nr - f_nr)
  else:
    can_m = set()
    for m_nr in range(math.floor(f_nr), min_nr, -1):
      if m_nr in nr_m_dict:
        can_m = can_m.union(nr_m_dict[m_nr])
    rnd_m = random.choice(list(can_m))
    sel_m.add(rnd_m)
    m_nr = m_nr_dict[rnd_m]
    nr_m_dict[m_nr].remove(rnd_m)
    m_minus_f += (m_nr - f_nr)
assert len(sel_m) == exp_m
avg_m = 0.0
for m in sel_m:
  avg_m += m_nr_dict[m]
avg_m /= exp_m
print('male avg=%.2f' % (avg_m))

m_set = sel_m
all_ratings = all_ratings[all_ratings.u.isin(m_set.union(f_set))]
user_list = sorted(all_ratings.u.unique())
user_dict = {u: j for j, u in enumerate(user_list)}
item_list = sorted(all_ratings.i.unique())
item_dict = {i: j for j, i in enumerate(item_list)}
all_ratings.u = all_ratings.u.map(user_dict)
all_ratings.i = all_ratings.i.map(item_dict)

num_users = all_ratings.u.unique().shape[0]
num_items = all_ratings.i.unique().shape[0]
all_ratings = all_ratings.reindex(np.random.permutation(all_ratings.index))
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

user_attr = pd.read_csv(in_user_file, sep='|', names=['u', 'a', 'g', 'o', 'c'])
user_attr.u = user_attr.u - 1
user_attr = user_attr[user_attr.u.isin(m_set.union(f_set))]
user_attr.u = user_attr.u.map(user_dict)
kwargs['columns'] = ['u', 'g']
user_attr.to_csv(out_user_file, **kwargs)

user_attr = user_attr.set_index('u').to_dict()['g']
m_train_size = 0
f_train_size = 0
for row in train_ratings.itertuples():
  attr = user_attr[row.u]
  if attr == 'M':
    m_train_size += 1
  elif attr == 'F':
    f_train_size += 1
  else:
    raise Exception('unknown attribute %s' % (attr))
d_train_size = abs(m_train_size - f_train_size)
print('train m=%d f=%d d=%d' % (m_train_size, f_train_size, d_train_size))













