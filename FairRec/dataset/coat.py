from os import path

import numpy as np
import os

dnld_dir = path.expanduser('~/Downloads')
in_dir = path.join(dnld_dir, 'coat')
in_train_file = path.join(in_dir, 'train.ascii')
in_test_file = path.join(in_dir, 'test.ascii')
feat_dir = path.join(in_dir, 'user_item_features')
user_feat_file = path.join(feat_dir, 'user_features.ascii')

cur_dir = path.dirname(path.realpath(__file__))
dataset = path.basename(__file__).split('.')[0]
out_dir = path.join(cur_dir, dataset)
if not path.exists(out_dir):
  os.makedirs(out_dir)
out_train_file = path.join(out_dir, 'train.data')
out_test_file = path.join(out_dir, 'test.data')
out_stat_file = path.join(out_dir, 'data.stat')
out_user_file = path.join(out_dir, 'user.attr')

train_mat = np.loadtxt(in_train_file)
num_users, num_items = train_mat.shape
train_users, train_items = train_mat.nonzero()
train_size = np.count_nonzero(train_mat)

test_mat = np.loadtxt(in_test_file)
test_users = []
test_items = []
test_size = 0
for u, i in zip(*test_mat.nonzero()):
  if train_mat[u, i] > 0:
    continue
  test_users.append(u)
  test_items.append(i)
  test_size += 1

num_males = 0
num_females = 0
user_feat_mat = np.loadtxt(user_feat_file)
with open(out_user_file, 'w') as fout:
  for u in range(num_users):
    if user_feat_mat[u, 0] == 1:
      assert user_feat_mat[u, 1] == 0
      attr = 'M'
      num_males += 1
    else:
      assert user_feat_mat[u, 1] == 1
      attr = 'F'
      num_females += 1
    fout.write('%d\t%s\n' % (u, attr))

with open(out_stat_file, 'w') as fout:
  fout.write('#users=%d\n' % (num_users))
  fout.write('#items=%d\n' % (num_items))
  fout.write('#train=%d\n' % (train_size))
  fout.write('#test=%d\n' % (test_size))
  fout.write('#male=%d\n' % (num_males))
  fout.write('#female=%d\n' % (num_females))

with open(out_train_file, 'w') as fout:
  for u, i in zip(train_users, train_items):
    fout.write('%d\t%d\t%d\n' % (u, i, train_mat[u, i]))

with open(out_test_file, 'w') as fout:
  for u, i in zip(test_users, test_items):
    fout.write('%d\t%d\t%d\n' % (u, i, test_mat[u, i]))

all_mat = train_mat + test_mat
all_users, all_items = all_mat.nonzero()
num_all = np.count_nonzero(all_mat)
train_size = int(0.8 * num_all) + 1
indexes = np.random.permutation(num_all)
with open(out_train_file, 'w') as fout:
  for j in range(train_size):
    j = indexes[j]
    u = all_users[j]
    i = all_items[j]
    if train_mat[u, i] > 0:
      fout.write('%d\t%d\t%d\n' % (u, i, train_mat[u, i]))
    else:
      assert test_mat[u, i] > 0
      fout.write('%d\t%d\t%d\n' % (u, i, test_mat[u, i]))
with open(out_test_file, 'w') as fout:
  for j in range(test_size, num_all):
    j = indexes[j]
    u = all_users[j]
    i = all_items[j]
    if train_mat[u, i] > 0:
      fout.write('%d\t%d\t%d\n' % (u, i, train_mat[u, i]))
    else:
      assert test_mat[u, i] > 0
      fout.write('%d\t%d\t%d\n' % (u, i, test_mat[u, i]))








