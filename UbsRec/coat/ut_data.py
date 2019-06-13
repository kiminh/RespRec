from os import path

import numpy as np
import pandas as pd

def parse_index(i_str):
  if ':' in i_str:
    i_list = []
    for i_range in i_str.split(','):
      i_range = i_range.split(':')
      start = int(i_range[0])
      stop = int(i_range[1])
      for i in range(start, stop):
        i_list.append(i)
  else:
    i_list = eval(i_str)
  return i_list

class IpsData(object):
  def __init__(self, inputs, outputs, weights):
    self._inputs = inputs
    self._outputs = outputs
    self._weights = weights
    self._nnz_input = inputs.shape[1]
    self._tot_input = inputs.max() + 1
    self._data_size = outputs.shape[0]
    self._start = 0

  @property
  def inputs(self):
    return self._inputs
  
  @property
  def outputs(self):
    return self._outputs
  
  @property
  def weights(self):
    return self._weights

  @property
  def nnz_input(self):
    return self._nnz_input

  @property
  def tot_input(self):
    return self._tot_input

  @property
  def data_size(self):
    return self._data_size

  def shuffle_data(self):
    indexes = np.arange(self.data_size)
    indexes = np.random.permutation(indexes)
    self._inputs = self.inputs[indexes]
    self._outputs = self.outputs[indexes]
    self._weights = self.weights[indexes]

  def next_batch(self, batch_size):
    stop = self._start + batch_size
    if stop < self.data_size:
      _inputs = self.inputs[self._start:stop, :]
      _outputs = self.outputs[self._start:stop]
      _weights = self.weights[self._start:stop]
    else:
      stop = stop - self.data_size
      _inputs = np.concatenate((self.inputs[self._start:self.data_size, :],
                                self.inputs[:stop, :]))
      _outputs = np.concatenate((self.outputs[self._start:self.data_size],
                                 self.outputs[:stop]))
      _weights = np.concatenate((self.weights[self._start:self.data_size],
                                 self.weights[:stop]))
    self._start = stop
    return _inputs, _outputs, _weights

class LtrData(object):
  def __init__(self, inputs, outputs, disc_inputs, cont_inputs):
    self._inputs = inputs
    self._outputs = outputs
    self._disc_inputs = disc_inputs
    self._cont_inputs = cont_inputs
    self._nnz_input = inputs.shape[1]
    self._tot_input = inputs.max() + 1
    self._data_size = outputs.shape[0]
    self._nnz_disc_input = disc_inputs.shape[1]
    self._tot_disc_input = disc_inputs.max() + 1
    self._tot_cont_input = cont_inputs.shape[1]
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
  def cont_inputs(self):
    return self._cont_inputs

  @property
  def nnz_input(self):
    return self._nnz_input

  @property
  def tot_input(self):
    return self._tot_input

  @property
  def data_size(self):
    return self._data_size

  @property
  def nnz_disc_input(self):
    return self._nnz_disc_input

  @property
  def tot_disc_input(self):
    return self._tot_disc_input

  @property
  def tot_cont_input(self):
    return self._tot_cont_input

  def shuffle_data(self):
    indexes = np.arange(self.data_size)
    indexes = np.random.permutation(indexes)
    self._inputs = self.inputs[indexes]
    self._outputs = self.outputs[indexes]
    self._disc_inputs = self.disc_inputs[indexes]
    self._cont_inputs = self.cont_inputs[indexes]

  def next_batch(self, batch_size):
    stop = self._start + batch_size
    if stop < self.data_size:
      _inputs = self.inputs[self._start:stop, :]
      _outputs = self.outputs[self._start:stop]
      _disc_inputs = self.disc_inputs[self._start:stop, :]
      _cont_inputs = self.cont_inputs[self._start:stop, :]
    else:
      stop = stop - self.data_size
      _inputs = np.concatenate((self.inputs[self._start:self.data_size, :],
                                  self.inputs[:stop, :]))
      _outputs = np.concatenate((self.outputs[self._start:self.data_size],
                                 self.outputs[:stop]))
      _disc_inputs = np.concatenate((self.disc_inputs[self._start:self.data_size, :],
                                     self.disc_inputs[:stop, :]))
      _cont_inputs = np.concatenate((self.cont_inputs[self._start:self.data_size, :],
                                     self.cont_inputs[:stop, :]))
    self._start = stop
    return _inputs, _outputs, _disc_inputs, _cont_inputs

def get_ips_data(tf_flags):
  def _get_ips_data(file_base):
    data_file = file_base + '.dta'
    data_df = pd.read_csv(data_file, sep='\t', header=None)
    inputs = data_df.values[:, i_input].astype(int)
    outputs = data_df.values[:, -1].astype(float)

    weight_file = file_base + '.wt'
    weight_df = pd.read_csv(weight_file, sep='\t', header=None)
    weights = weight_df.values[:, 0].astype(float)
    weights = 1.0 / weights
  
    data_set = IpsData(inputs, outputs, weights)
    return data_set

  data_dir = tf_flags.data_dir
  i_input = parse_index(tf_flags.i_input)

  train_base = path.join(data_dir, 'train')
  valid_base = path.join(data_dir, 'valid')
  test_base = path.join(data_dir, 'test')
  train_set = _get_ips_data(train_base)
  valid_set = _get_ips_data(valid_base)
  test_set = _get_ips_data(test_base)
  return train_set, valid_set, test_set

def get_ltr_data(tf_flags):
  def _get_ltr_data(data_file):
    data_df = pd.read_csv(data_file, sep='\t', header=None)
    inputs = data_df.values[:, i_input].astype(int)
    outputs = data_df.values[:, -1].astype(float)
    disc_inputs = data_df.values[:, i_disc_input].astype(int)
    cont_inputs = data_df.values[:, i_cont_input].astype(float)
    min_cont = cont_inputs.min(axis=0)
    max_cont = cont_inputs.max(axis=0)
    cont_inputs = (cont_inputs - min_cont) / (max_cont - min_cont)
    # input([disc_inputs.shape, cont_inputs.shape])
    data_set = LtrData(inputs, outputs, disc_inputs, cont_inputs)
    return data_set

  data_dir = tf_flags.data_dir
  i_input = parse_index(tf_flags.i_input)
  i_disc_input = parse_index(tf_flags.i_disc_input)
  i_cont_input = parse_index(tf_flags.i_cont_input)
  # input([i_disc_input, i_cont_input])

  train_file = path.join(data_dir, 'train.dta')
  valid_file = path.join(data_dir, 'valid.dta')
  test_file = path.join(data_dir, 'test.dta')
  train_set = _get_ltr_data(train_file)
  valid_set = _get_ltr_data(valid_file)
  test_set = _get_ltr_data(test_file)
  return train_set, valid_set, test_set

