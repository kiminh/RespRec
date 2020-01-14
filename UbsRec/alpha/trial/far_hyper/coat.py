from os import path

import numpy as np
import pandas as pd

class Datasets(object):
  def __init__(self, train=None, valid=None, test=None):
    self.train = train
    self.valid = valid
    self.test = test

  @staticmethod
  def from_list(list_of_datasets):
    train, valid, test = None, None, None
    assert len(list_of_datasets) == 3
    train = list_of_datasets[0]
    valid = list_of_datasets[1]
    test = list_of_datasets[-1]
    return Datasets(train, valid, test)

class Dataset(object):
  def __init__(self, data, target):
    self._data = data
    self._target = target

  @property
  def data(self):
    return self._data
  
  @property
  def target(self):
    return self._target

  @property
  def num_examples(self):
    return self.data.shape[0]

  def create_supplier(self, x, y, batch_size=None):
    if batch_size:
      raise Exception('to implement')
    else:
      def _supplier(step=0):
          return {x: self.data, y: self.target}
    return _supplier

def get_dataset(data_file):
  ratings = pd.read_csv(data_file, sep='\t', header=None)
  data = ratings.values[:, 1:].astype(int)
  target = np.squeeze(ratings.values[:, :1]).astype(float)
  dataset = Dataset(data, target)
  return dataset

def get_datasets(data_dir):
  train_file = path.join(data_dir, 'train.dta')
  valid_file = path.join(data_dir, 'valid.dta')
  test_file = path.join(data_dir, 'test.dta')
  train = get_dataset(train_file)
  valid = get_dataset(valid_file)
  test = get_dataset(test_file)
  res = [train, valid, test]
  return Datasets.from_list(res)
