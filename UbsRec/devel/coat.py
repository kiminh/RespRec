from os import path

import numpy as np
import pandas as pd

class Dataset(object):
  def __init__(self, features, ratings):
    self._features = features
    self._ratings = ratings
    self._data_size, self._n_feature = features.shape
    self._t_feature = (features.max(axis=0) + 1).sum()
    self._start = 0

  @property
  def features(self):
    return self._features
  
  @property
  def ratings(self):
    return self._ratings

  @property
  def data_size(self):
    return self._data_size

  @property
  def n_feature(self):
    return self._n_feature
  
  @property
  def t_feature(self):
    return self._t_feature

  def shuffle_in_unison(self):
    rnd_state = np.random.get_state()
    np.random.shuffle(self.features)
    np.random.set_state(rnd_state)
    np.random.shuffle(self.ratings)

  def next_batch(self, batch_size):
    stop = self._start + batch_size
    if stop < self.data_size:
      _features = self.features[self._start:stop, :]
      _ratings = self.ratings[self._start:stop]
    else:
      stop = stop - self.data_size
      _features = np.concatenate((self.features[self._start:self.data_size, :],
                                  self.features[:stop, :]))
      _ratings = np.concatenate((self.ratings[self._start:self.data_size],
                                 self.ratings[:stop]))
    # input([self._start, stop])
    self._start = stop
    return _features, _ratings

def get_dataset(data_file):
  ratings = pd.read_csv(data_file, sep='\t', header=None)
  features = ratings.values[:, 1:].astype(int)
  ratings = np.squeeze(ratings.values[:, :1]).astype(float)
  dataset = Dataset(features, ratings)
  return dataset

def get_datasets(data_dir):
  train_file = path.join(data_dir, 'train.dta')
  valid_file = path.join(data_dir, 'valid.dta')
  test_file = path.join(data_dir, 'test.dta')
  train_set = get_dataset(train_file)
  valid_set = get_dataset(valid_file)
  test_set = get_dataset(test_file)
  return train_set, valid_set, test_set


'''
class Dataset(object):
  def __init__(self, base_dir):
    self.offset = 2
    self.train_file = base_dir + '.train'
    self.valid_file = base_dir + '.valid'
    self.test_file = base_dir + '.test'
    self.num_features = self.count_num_feature()
    self.nnz_features = self.count_nnz_feature()
    self.train_data, self.valid_data, self.test_data = self.load_data()
    self.train_size = len(self.train_data[fkey])

    user_feat_file = base_dir + '.user'
    item_feat_file = base_dir + '.item'
    self.user_features = self.load_feature(user_feat_file)
    self.item_features = self.load_feature(item_feat_file)

    self.num_users = len(self.user_features)
    self.num_items = len(self.item_features)

  def count_num_feature(self):
    features = set()
    fin = open(self.train_file)
    line = fin.readline()
    while line:
      fields = line.strip().split()
      for feature in fields[1:]:
        features.add(feature)
      line = fin.readline()
    fin.close()
    return len(features)

  def count_nnz_feature(self):
    offset = self.offset
    fin = open(self.train_file)
    line = fin.readline()
    fields = line.strip().split()
    fin.close()
    return len(fields) - offset

  def load_data(self):
    train_data = self.read_data(self.train_file)
    valid_data = self.read_data(self.valid_file)
    test_data = self.read_data(self.test_file)
    return train_data, valid_data, test_data

  def read_data(self, file):
    offset = self.offset
    features = []
    ratings = []
    fin = open(file)
    line = fin.readline()
    while line:
      fields = line.strip().split()
      features.append([int(feature) for feature in fields[offset:]])
      ratings.append(1.0 * float(fields[0]))
      line = fin.readline()
    fin.close()
    self.min_value = min(ratings)
    self.max_value = max(ratings)
    data = {fkey: features, rkey: ratings}
    return data

  def load_feature(self, file):
    features = []
    fin = open(file)
    line = fin.readline()
    while line:
      features.append([int(feature) for feature in line.strip().split()])
      line = fin.readline()
    return features
'''
