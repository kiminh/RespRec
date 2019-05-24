from os import path

import numpy as np
import pandas as pd

class Dataset(object):
  def __init__(self, inputs, outputs, disc_inputs):
    self._inputs = inputs
    self._outputs = outputs
    self._disc_inputs = disc_inputs
    self._nnz_input = inputs.shape[1]
    self._nnz_disc_input = disc_inputs.shape[1]
    self._data_size = outputs.shape[0]
    self._tot_input = inputs.max() + 1
    self._tot_disc_input = disc_inputs.max() + 1
    self._start = 0

  @property
  def inputs(self):
    return self._inputs
  
  @property
  def outputs(self):
    return self._outputs

  @property
  def disc_inputs(self):
    return self._disc_inputs

  @property
  def nnz_input(self):
    return self._nnz_input

  @property
  def nnz_disc_input(self):
    return self._nnz_disc_input

  @property
  def data_size(self):
    return self._data_size
  
  @property
  def tot_input(self):
    return self._tot_input

  @property
  def tot_disc_input(self):
    return self._tot_disc_input

  def shuffle_data(self):
    indexes = np.arange(self.data_size)
    indexes = np.random.permutation(indexes)
    self._inputs = self.inputs[indexes]
    self._outputs = self.outputs[indexes]
    self._disc_inputs = self.disc_inputs[indexes]

  def next_batch(self, batch_size):
    stop = self._start + batch_size
    if stop < self.data_size:
      _inputs = self.inputs[self._start:stop, :]
      _outputs = self.outputs[self._start:stop]
      _disc_inputs = self.disc_inputs[self._start:stop, :]
    else:
      stop = stop - self.data_size
      _inputs = np.concatenate((self.inputs[self._start:self.data_size, :],
                                  self.inputs[:stop, :]))
      _outputs = np.concatenate((self.outputs[self._start:self.data_size],
                                 self.outputs[:stop]))
      _disc_inputs = np.concatenate((self.disc_inputs[self._start:self.data_size, :],
                                     self.disc_inputs[:stop, :]))
    # input([self._start, stop])
    self._start = stop
    return _inputs, _outputs, _disc_inputs

def get_dataset(data_file):
  dataset = pd.read_csv(data_file, sep='\t', header=None)
  inputs = dataset.values[:, 0:2].astype(int)
  outputs = dataset.values[:, -1].astype(float)
  disc_inputs = dataset.values[:, 0:-1].astype(int)
  dataset = Dataset(inputs, outputs, disc_inputs)
  return dataset

def get_datasets(data_dir):
  train_file = path.join(data_dir, 'train.dta')
  valid_file = path.join(data_dir, 'valid.dta')
  test_file = path.join(data_dir, 'test.dta')
  train_set = get_dataset(train_file)
  valid_set = get_dataset(valid_file)
  test_set = get_dataset(test_file)
  return train_set, valid_set, test_set
