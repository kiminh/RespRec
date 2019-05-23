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
